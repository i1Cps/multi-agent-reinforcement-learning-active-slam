import numpy as np
from pathlib import Path
import torch as T
import time
import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty

from multi_robot_active_slam_interfaces.srv import StepEnv, ResetEnv
from multi_robot_active_slam_learning.learning.mappo.mappo import MAPPO
from multi_robot_active_slam_learning.learning.mappo.memory import (
    MAPPOMemory,
)
from multi_robot_active_slam_learning.common import utilities as util
from multi_robot_active_slam_learning.common.settings import (
    MAX_STEPS,
    NUMBER_OF_ROBOTS,
    ROBOT_OBSERVATION_SPACE,
    ROBOT_ACTION_SPACE,
    MAX_CONTINUOUS_ACTIONS,
    GAMMA_MAPPO,
    N_EPOCHS,
    GAE_LAMBDA,
    ENTROPHY_COEFFICIENT,
    POLICY_CLIP,
    BETA_MAPPO,
    ALPHA_MAPPO,
    ACTOR_MAPPO_FC1,
    ACTOR_MAPPO_FC2,
    CRITIC_MAPPO_FC1,
    CRITIC_MAPPO_FC2,
    BATCH_SIZE_MAPPO,
    MINI_BATCHES_MAPPO,
    TRAINING_EPISODES,
    FRAME_BUFFER_DEPTH,
    FRAME_BUFFER_SKIP,
)


class LearningMAPPO(Node):
    def __init__(self):
        super().__init__("learning_mappo")

        self.number_of_agents = NUMBER_OF_ROBOTS
        self.stack_depth = FRAME_BUFFER_DEPTH  # Number of frames to stack
        self.frame_skip = FRAME_BUFFER_SKIP

        # Initializing state and action dimensions for each agent
        self.actor_dims = [
            ROBOT_OBSERVATION_SPACE[0] * self.stack_depth
            for _ in range(self.number_of_agents)
        ]
        self.number_of_actions = [
            ROBOT_ACTION_SPACE[0] for _ in range(self.number_of_agents)
        ]
        self.critic_dims = sum(self.actor_dims)

        # Creating frame buffers for each agent, initialised to 3
        self.frame_buffers = [
            np.full(
                (self.stack_depth * ROBOT_OBSERVATION_SPACE[0]), 3, dtype=np.float32
            )
            for _ in range(self.number_of_agents)
        ]

        self.model_agents = MAPPO(
            actor_dims=self.actor_dims,
            critic_dims=self.critic_dims,
            n_agents=self.number_of_agents,
            n_actions=self.number_of_actions,
            n_epochs=N_EPOCHS,
            entropy_coefficient=ENTROPHY_COEFFICIENT,
            gae_lambda=GAE_LAMBDA,
            policy_clip=POLICY_CLIP,
            gamma=GAMMA_MAPPO,
            alpha=ALPHA_MAPPO,
            beta=BETA_MAPPO,
            actor_fc1=ACTOR_MAPPO_FC1,
            actor_fc2=ACTOR_MAPPO_FC2,
            critic_fc1=CRITIC_MAPPO_FC1,
            critic_fc2=CRITIC_MAPPO_FC2,
            checkpoint_dir="models/",
            scenario="multi_agent_reinforcement_learning_active_slam",
        )

        self.memory = MAPPOMemory(
            batch_size=MINI_BATCHES_MAPPO,
            T=BATCH_SIZE_MAPPO,
            n_agents=self.number_of_agents,
            critic_dims=self.critic_dims,
            actor_dims=self.actor_dims,
            n_actions=self.number_of_actions,
        )

        self.total_episodes = TRAINING_EPISODES
        self.max_steps = MAX_STEPS
        self.score_history, self.step_history = [], []
        self.best_score = -np.Inf
        self.current_frame = 0
        self.total_steps = 0
        self.episode = 1
        self.traj_length = 0
        self.model_step_time = 0.01
        self.score = 0

        # --------------------- Clients ---------------------------#

        self.gazebo_pause = self.create_client(Empty, "/pause_physics")
        self.gazebo_unpause = self.create_client(Empty, "/unpause_physics")
        self.environment_step_client = self.create_client(StepEnv, "/environment_step")
        self.reset_environment_client = self.create_client(
            ResetEnv, "/reset_environment_rl"
        )

        # Check for GPU availability
        self.get_logger().info(
            "GPU AVAILABLE" if T.cuda.is_available() else "GPU UNAVAILABLE"
        )

        # Start Reinforcement Learning
        self.get_logger().info("Starting the learning loop")
        self.start()

    def update_frame_buffers(self, observations):
        for i, obs in enumerate(observations):
            self.frame_buffers[i] = np.roll(
                self.frame_buffers[i], -ROBOT_OBSERVATION_SPACE[0]
            )
            self.frame_buffers[i][-ROBOT_OBSERVATION_SPACE[0] :] = obs

    def get_stacked_observations(self, agent_idx):
        return self.frame_buffers[agent_idx]

    def clear_frame_buffers(self):
        for i in range(len(self.frame_buffers)):
            self.frame_buffers[i].fill(3)

    def action_adapter(self, action, max_action):
        return 2 * (action - 0.5) * max_action

    # Main learning loop
    def start(self):
        util.pause_simulation(self)
        while self.total_steps < MAX_STEPS:
            obs = util.reset(self)
            self.clear_frame_buffers()
            self.update_frame_buffers(obs)
            raw_stacked_observations = [
                self.get_stacked_observations(i) for i in range(self.number_of_agents)
            ]

            global_stacked_observations = np.concatenate(raw_stacked_observations)
            done = False
            self.score = 0
            self.current_frame = 0

            time.sleep(0.5)
            util.unpause_simulation(self)
            while not done:
                self.current_frame += 1
                if self.current_frame % self.frame_skip == 0:
                    actions, probs = self.model_agents.choose_actions(
                        raw_stacked_observations
                    )
                    adapted_actions = self.action_adapter(
                        actions, MAX_CONTINUOUS_ACTIONS
                    )
                    next_obs, reward, terminals, truncated = util.step(
                        self, adapted_actions
                    )

                    self.total_steps += 1
                    self.traj_length += 1

                    done = any(d or t for d, t in zip(terminals, truncated))
                    self.score = sum(reward)

                    self.update_frame_buffers(next_obs)
                    next_raw_stacked_observations = [
                        self.get_stacked_observations(i)
                        for i in range(self.number_of_agents)
                    ]

                    next_global_stacked_observations = np.concatenate(
                        next_raw_stacked_observations
                    )

                    self.memory.store_memory(
                        raw_obs=raw_stacked_observations,
                        state=global_stacked_observations,
                        action=actions,
                        prob=probs,
                        reward=reward,
                        next_raw_obs=next_raw_stacked_observations,
                        next_state=next_global_stacked_observations,
                        terminal=done,
                    )

                    if self.traj_length % BATCH_SIZE_MAPPO == 0:
                        util.pause_simulation(self)
                        self.model_agents.learn(self.memory)
                        util.unpause_simulation(self)
                        self.traj_length = 0
                        self.memory.clear_memory()
                    global_stacked_observations = next_global_stacked_observations
                    raw_stacked_observations = next_raw_stacked_observations

                time.sleep(self.model_step_time)

            util.pause_simulation(self)
            self.score_history.append(self.score)
            self.step_history.append(self.total_steps)
            avg_score = np.mean(self.score_history[-100:])
            print(
                f"Active SLAM Episode {self.episode} total steps {self.total_steps}"
                f" score {self.score} avg score {avg_score :.1f}"
            )
            self.episode += 1

        mappo_scores_path = Path(
            "src/multi_robot_active_slam_learning/data/raw_data/mappo_scores.npy"
        )

        mappo_steps_path = Path(
            "src/multi_robot_active_slam_learning/data/raw_data/mappo_steps.npy"
        )
        mappo_scores_path.mkdir(parents=True, exist_ok=True)
        mappo_steps_path.mkdir(parents=True, exist_ok=True)
        np.save(
            mappo_scores_path,
            np.array(self.score_history),
        )
        np.save(
            mappo_steps_path,
            np.array(self.step_history),
        )


def main():
    rclpy.init()
    learning_mappo = LearningMAPPO()
    rclpy.spin(learning_mappo)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
