import numpy as np
import torch as T
import rclpy
from rclpy.node import Node

from multi_robot_active_slam_interfaces.srv import StepEnv, ResetEnv
from multi_robot_active_slam_learning.learning.mappo.mappo import MAPPO
from multi_robot_active_slam_learning.learning.mappo.memory import (
    PPOMemory,
)
from multi_robot_active_slam_learning.learning.maddpg.utils import plot_learning_curve
from multi_robot_active_slam_learning.common import utilities as util
from multi_robot_active_slam_learning.common.settings import (
    NUMBER_OF_ROBOTS,
    ROBOT_OBSERVATION_SPACE,
    ROBOT_ACTION_SPACE,
    MAX_CONTINUOUS_ACTIONS,
    TAU_MADDPG,
    GAMMA_MADDPG,
    BETA_MADDPG,
    ALPHA_MADDPG,
    ACTOR_MADDPG_FC1,
    ACTOR_MADDPG_FC2,
    CRITIC_MADDPG_FC1,
    CRITIC_MADDPG_FC2,
    MAX_MEMORY_SIZE,
    BATCH_SIZE,
    TRAINING_EPISODES,
    RANDOM_STEPS,
    FRAME_BUFFER_DEPTH,
    FRAME_BUFFER_SKIP,
)


class LearningMAPPO(Node):
    def __init__(self):
        super().__init__("learning_mappo")

        self.number_of_agents = NUMBER_OF_ROBOTS
        self.stack_depth = FRAME_BUFFER_DEPTH  # Number of frames to stack
        self.frame_skip = FRAME_BUFFER_SKIP
        self.current_frame = 0

        # Initializing state and action dimensions for each agent
        self.actor_dims = [
            ROBOT_OBSERVATION_SPACE[0] for _ in range(self.number_of_agents)
        ]
        self.number_of_actions = [
            ROBOT_ACTION_SPACE[0] for _ in range(self.number_of_agents)
        ]
        self.critic_dims = self.calculate_critic_input_dims()

        # Creating frame buffers for each agent
        self.frame_buffers = [
            np.zeros((dim * self.stack_depth,)) for dim in self.actor_dims
        ]

        self.model_agents = MAPPO(
            actor_dims=self.actor_dims,
            critic_dims=self.critic_dims,
            n_agents=self.number_of_agents,
            n_actions=self.number_of_actions,
            gamma=GAMMA_MADDPG,
            alpha=ALPHA_MADDPG,
            beta=BETA_MADDPG,
            tau=TAU_MADDPG,
            actor_dims_fc1=ACTOR_MADDPG_FC1,
            actor_dims_fc2=ACTOR_MADDPG_FC2,
            critic_dims_fc1=CRITIC_MADDPG_FC1,
            critic_dims_fc2=CRITIC_MADDPG_FC2,
            max_action=MAX_CONTINUOUS_ACTIONS,
            min_action=MAX_CONTINUOUS_ACTIONS * -1,
            stacked_frames=self.stack_depth,
        )

        self.critic_dims = sum(self.actor_dims)

        self.memory = MultiAgentReplayBuffer(
            max_size=MAX_MEMORY_SIZE,
            critic_dims=self.critic_dims * self.stack_depth,
            actor_dims=[dim * self.stack_depth for dim in self.actor_dims],
            n_actions=self.number_of_actions,
            n_agents=self.number_of_agents,
            batch_size=BATCH_SIZE,
        )

        self.total_steps = 0
        self.episode = 0

        self.total_episodes = TRAINING_EPISODES
        self.max_steps = 1_500_000
        self.score_history = []
        self.best_score = 0

        # --------------------- Clients ---------------------------#

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

    def calculate_critic_input_dims(self):
        stacked_actor_dims = ROBOT_OBSERVATION_SPACE[0] * self.stack_depth
        total_state_dims = stacked_actor_dims * self.number_of_agents
        total_action_dims = ROBOT_ACTION_SPACE[0] * self.number_of_agents
        return total_state_dims + total_action_dims

    def update_frame_buffers(self, observations):
        if self.current_frame % self.frame_skip == 0:
            for i, obs in enumerate(observations):
                self.frame_buffers[i] = np.roll(
                    self.frame_buffers[i], -self.actor_dims[i]
                )
                self.frame_buffers[i][-self.actor_dims[i] :] = obs

    def get_stacked_observations(self, agent_idx):
        # Retrieve and stack the observations for a specific agent
        return self.frame_buffers[agent_idx].copy()

    # Main learning loop
    def start(self):
        self.total_steps = 20000
        while self.total_steps < self.max_steps:
            observations = util.reset(self)
            self.update_frame_buffers(observations)
            raw_stacked_observations = [
                self.get_stacked_observations(i) for i in range(self.number_of_agents)
            ]
            global_stacked_observations = np.concatenate(raw_stacked_observations)
            done = [False] * self.number_of_agents
            score = 0
            self.current_frame = 0
            while not any(done):
                self.current_frame += 1
                if self.current_frame % self.frame_skip == 0:
                    if self.total_steps < RANDOM_STEPS:
                        action = self.model_agents.choose_random_actions()
                    else:
                        action = self.model_agents.choose_actions(
                            raw_stacked_observations
                        )

                    next_obs, reward, terminals, truncated = util.step(self, action)

                    done = [d or t for d, t in zip(terminals, truncated)]
                    self.update_frame_buffers(next_obs)
                    next_raw_stacked_observations = [
                        self.get_stacked_observations(i)
                        for i in range(self.number_of_agents)
                    ]
                    next_global_stacked_observations = np.concatenate(
                        next_raw_stacked_observations
                    )

                    self.memory.store_transition(
                        raw_obs=raw_stacked_observations,
                        state=global_stacked_observations,
                        action=action,
                        reward=reward,
                        next_raw_obs=next_raw_stacked_observations,
                        next_state=next_global_stacked_observations,
                        done=terminals or truncated,
                    )

                    score += sum(reward)
                    observations = next_obs
                    global_stacked_observations = next_global_stacked_observations

                self.total_steps += 1
            self.finish_episode(score)

        x = [i + 1 for i in range(self.total_episodes)]
        filename = "./src/multi_robot_active_slam_learning/multi_robot_active_slam_learning/learning/mappo/plots/mappo.png"
        np.save("mappo_score.npy", np.array(self.score_history))
        np.save("mappo_steps.npy", np.array(x))
        plot_learning_curve(x, self.score_history, filename)
        # self.shutdown_nodes()

    # Handles end of episode (nice, clean and modular)
    def finish_episode(self, score):
        self.episode += 1
        # Calculate the robot avearage score
        self.score_history.append(score)
        # Average the last 100 recent scores
        avg_score = np.mean(self.score_history[-100:])
        if avg_score > self.best_score:
            self.best_score = avg_score
            self.model_agents.save_models()

        self.get_logger().info(
            "Episode: {}, score: {}, Average Score: {:.1f}".format(
                self.episode, score, avg_score
            )
        )

    def obs_list_to_state_vector(self, observation):
        state = np.array([])
        for obs in observation:
            state = np.concatenate([state, obs])
        return state


def main():
    rclpy.init()
    learning_mappo = LearningMAPPO()
    rclpy.spin(learning_mappo)
    # learning_mappo.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
