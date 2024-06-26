from pathlib import Path
import time
import torch as T
import rclpy
from rclpy.node import Node

import numpy as np

from std_srvs.srv import Empty

from multi_robot_active_slam_interfaces.srv import StepEnv, ResetEnv
from multi_robot_active_slam_learning.learning.maddpg.maddpg import MADDPG
from multi_robot_active_slam_learning.learning.maddpg.memory import (
    MultiAgentReplayBuffer,
)

from multi_robot_active_slam_learning.common import utilities as util
from multi_robot_active_slam_learning.common.settings import (
    MODEL_PATH,
    ACTOR_LEARNING_RATE_MADDPG,
    CRITIC_LEARNING_RATE_MADDPG,
    BATCH_SIZE_MADDPG,
    FRAME_BUFFER_DEPTH,
    FRAME_BUFFER_SKIP,
    TAU,
    GAMMA_MADDPG,
    ACTOR_MADDPG_FC1,
    ACTOR_MADDPG_FC2,
    CRITIC_MADDPG_FC1,
    CRITIC_MADDPG_FC2,
    MAX_MEMORY_SIZE,
    RANDOM_STEPS,
    MAX_CONTINUOUS_ACTIONS,
    ROBOT_OBSERVATION_SPACE,
    ROBOT_ACTION_SPACE,
    MAX_SCAN_DISTANCE,
    TRAINING_EPISODES,
    TRAINING_STEPS,
    NUMBER_OF_ROBOTS,
    LOAD_MODEL,
)


class LearningMADDPG(Node):
    def __init__(self):
        super().__init__("learning_maddpg")

        self.initialise_parameters()
        self.model_agents = self.initialise_model()
        self.memory = self.initialise_memory()
        self.initialise_clients()

        # Check for GPU availability
        self.get_logger().info(
            "GPU AVAILABLE" if T.cuda.is_available() else "WARNING GPU UNAVAILABLE"
        )

        # Start Reinforcement Learning
        self.start_training()
        self.end_training()
        # Save data
        self.save_training_data()

    def initialise_parameters(self):
        pass

        self.episode_number = 0
        self.total_steps = 0

        self.score_history = []
        self.step_history = []
        self.goal_history = []
        self.collision_history = []
        self.best_score = -np.Infinity

        self.training_start_time = time.perf_counter()
        self.training_episodes = TRAINING_EPISODES
        self.training_steps = TRAINING_STEPS
        self.load_model = LOAD_MODEL

        # Create Directory in user system
        self.model_path = Path(MODEL_PATH)
        self.model_path.mkdir(parents=True, exist_ok=True)

        # Frame stacking
        self.stack_depth = FRAME_BUFFER_DEPTH
        self.frame_skip = FRAME_BUFFER_SKIP
        self.frame_buffers = [
            np.full(
                (self.stack_depth * ROBOT_OBSERVATION_SPACE),
                MAX_SCAN_DISTANCE,
                dtype=np.float32,
            )
            for robot in range(NUMBER_OF_ROBOTS)
        ]

        # A list of input state spaces for each agent actor network
        self.actor_state_dims = [
            ROBOT_OBSERVATION_SPACE * self.stack_depth
            for robot in range(NUMBER_OF_ROBOTS)
        ]

        # A list of each agents action space
        self.n_actions = [ROBOT_ACTION_SPACE for robot in range(NUMBER_OF_ROBOTS)]

        # MADDPG ~ The state space for the critic network is the sum of every actors state dimension (READ THE PAPER)
        self.critic_state_dims = sum(self.actor_state_dims)

        # MADDPG ~ The input dims for the critic network is the sum of actor states + sum of action spaces
        self.critic_input_dims = self.critic_state_dims + sum(self.n_actions)

    def initialise_model(self) -> MADDPG:
        model = MADDPG(
            actor_input_dims=self.actor_state_dims,
            critic_input_dims=self.critic_input_dims,
            n_agents=NUMBER_OF_ROBOTS,
            n_actions=self.n_actions,
            alpha=ACTOR_LEARNING_RATE_MADDPG,
            beta=CRITIC_LEARNING_RATE_MADDPG,
            gamma=GAMMA_MADDPG,
            tau=TAU,
            max_actions=MAX_CONTINUOUS_ACTIONS,
            min_actions=MAX_CONTINUOUS_ACTIONS * -1,
            actor_fc1=ACTOR_MADDPG_FC1,
            actor_fc2=ACTOR_MADDPG_FC2,
            critic_fc1=CRITIC_MADDPG_FC1,
            critic_fc2=CRITIC_MADDPG_FC2,
        )

        if self.load_model:
            model.load(self.model_path)
        return model

    def initialise_memory(self) -> MultiAgentReplayBuffer:
        return MultiAgentReplayBuffer(
            max_size=MAX_MEMORY_SIZE,
            critic_state_dims=self.critic_state_dims,
            actor_state_dims=self.actor_state_dims,
            n_actions=self.n_actions,
            n_agents=NUMBER_OF_ROBOTS,
            batch_size=BATCH_SIZE_MADDPG,
        )

    def initialise_clients(self) -> None:
        self.step_environment_client = self.create_client(StepEnv, "/step_environment")
        self.reset_environment_client = self.create_client(
            ResetEnv, "/reset_environment"
        )
        self.skip_environment_frame_client = self.create_client(
            Empty, "/skip_environment_frame"
        )

    # Main learning loop
    def start_training(self):
        self.get_logger().info("Starting the Reinforcement Learning")
        while self.total_steps < self.training_steps:
            # Reset episode
            observations = util.reset(self)

            # Prepare frame buffers
            for buffer in self.frame_buffers:
                buffer.fill(MAX_SCAN_DISTANCE)
            _ = self.update_frame_buffers(observations)

            current_frame = 0
            dones = [False] * NUMBER_OF_ROBOTS
            score = 0
            goals_found = 0
            actions = self.model_agents.choose_random_actions()

            while not any(dones):
                current_frame += 1
                if current_frame % self.frame_skip == 0:
                    # Choose actions
                    if self.total_steps < RANDOM_STEPS:
                        actions = self.model_agents.choose_random_actions()
                    else:
                        actions = self.model_agents.choose_actions(self.frame_buffers)

                    # Step the environment
                    next_obs, rewards, terminals, truncated, info = util.step(
                        self, actions
                    )

                    # Book keep goals found per episode
                    goals_found += info["goal_found"]

                    # Check for episode termination
                    dones = terminals or truncated
                    self.collision_history.append(int(any(terminals)))

                    # Store the current state of the buffers
                    current_frame_buffers = self.frame_buffers.copy()
                    # Concatenate the buffers for a global view ~ MADDPG critic network
                    global_current_frame_buffers = np.concatenate(
                        current_frame_buffers, axis=0
                    )

                    clipped_rewards = self.clip_rewards(rewards)

                    # Update the frame buffer with next_obs and store it too
                    next_frame_buffers = self.update_frame_buffers(next_obs)
                    # Concatenate the buffers for a global view ~ MADDPG critic network
                    global_next_frame_buffers = np.concatenate(
                        next_frame_buffers, axis=0
                    )

                    # Store the transition in a memory buffer for sampling
                    self.memory.store_transition(
                        single_obs=current_frame_buffers,
                        global_obs=global_current_frame_buffers,
                        actions=actions,
                        rewards=clipped_rewards,
                        next_single_obs=next_frame_buffers,
                        next_global_obs=global_next_frame_buffers,
                        dones=dones,
                    )

                    # Accumulate rewards per step for each episode
                    score += sum(rewards)
                    observations = next_obs
                    self.total_steps += 1
                else:
                    util.skip_frame(self)
                    # Learn ~ MADDPG is off-policy so agent can learn on skipped frames
                    # Try total_steps % 100 == 0
                    if self.total_steps >= RANDOM_STEPS:
                        self.model_agents.learn(self.memory)

            # Reset noise correlation per episode
            self.model_agents.reset_noise()

            # Book keep scores, goals and step history for plots
            self.score_history.append(score)
            self.goal_history.append(goals_found)
            self.step_history.append(self.total_steps)

            self.finish_episode(score, goals_found)

    # Handles end of episode (nice, clean and modular)
    def finish_episode(self, score, goals_found):
        self.episode_number += 1
        episode_finish_time_sec = time.perf_counter() - self.training_start_time
        episode_finish_time_min = episode_finish_time_sec / 60
        episode_finish_time_hour = episode_finish_time_min / 60
        avg_score = np.mean(self.score_history[-100:])
        if avg_score > self.best_score:
            self.best_score = avg_score
            self.model_agents.save(self.model_path)

        self.get_logger().info(
            "\nEpisode: {}, Steps: {}/{}, Training Time Elaspsed: {:.2f} \n Score: {:.2f}, Average Score: {:.2f}, Goals Found: {}".format(
                self.episode_number,
                self.total_steps,
                self.training_steps,
                episode_finish_time_min,
                score,
                avg_score,
                goals_found,
            )
        )

    def save_training_data(self):
        raw_data_dir = Path("src/multi_robot_active_slam_learning/raw_data")
        plot_dir = Path("src/multi_robot_active_slam_learning/plots")
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(parents=True, exist_ok=True)
        np.save(
            raw_data_dir / "maddpg_scores.npy",
            np.array(self.score_history),
        )
        np.save(
            raw_data_dir / "maddpg_steps.npy",
            np.array(self.step_history),
        )
        np.save(raw_data_dir / "maddpg_goals_found.npy", np.array(self.goal_history))
        np.save(raw_data_dir / "maddpg_collision.npy", np.array(self.collision_history))

        self.get_logger().info(
            "\n\n\nTraining has finished! raw data is available in the workspace src/training_data/raw_data/ "
        )

        # Plot the data
        util.plot_training_data(
            steps_file=raw_data_dir / "maddpg_steps.npy",
            scores_file=raw_data_dir / "madppg_scores.npy",
            goal_history_file=raw_data_dir / "maddpg_goals_found.npy",
            learning_plot_filename=plot_dir / "maddpg_learning_plot",
            goals_plot_filename=plot_dir / "maddpg_returns_plot",
            goals_title="maddpg goals found",
            learning_title="maddpg returns",
        )

    # Drastically increases performance
    def clip_rewards(self, rewards: list[float]) -> list[float]:
        clipped_rewards = []
        for reward in rewards:
            if reward < -10:
                clipped_rewards.append(-10)
            elif reward > 10:
                clipped_rewards.append(10)
            else:
                clipped_rewards.append(reward)
        return clipped_rewards

    def update_frame_buffers(self, observations):
        for agent_index, observation in enumerate(observations):
            self.frame_buffers[agent_index] = np.roll(
                self.frame_buffers[agent_index], ROBOT_OBSERVATION_SPACE
            )
            self.frame_buffers[agent_index][-ROBOT_OBSERVATION_SPACE:] = observation
        return self.frame_buffers

    def end_training(self):
        # Print results
        print(
            "\n\n\nResults: "
            + "\nGoals found: {}".format(sum(self.goal_history))
            + "\nCollisions:  {}".format(sum(self.collision_history))
            + "\nBest Score:  {:.2f}".format(self.best_score)
            + "\nTotal Time (hours): {:.2f}".format(
                ((time.perf_counter() - self.training_start_time) / 60) / 60,
            )
        )

        # Remind user of hyperparameters used
        print(
            "\n\nHyperparameters: "
            + "\nAlpha: {}".format(ALPHA_MADDPG)
            + "\nBeta:  {}".format(BETA_MADDPG)
            + "\nTau:   {}".format(TAU)
            + "\nGamma: {}".format(GAMMA_MADDPG)
            + "\nActor Fully Connected Dims:  {}".format(ACTOR_MADDPG_FC1)
            + "\nCritic Fully Connected Dims: {}".format(CRITIC_MADDPG_FC1)
            + "\nBatch Size: {}".format(BATCH_SIZE_MADDPG)
        )


def main():
    rclpy.init()
    learning_maddpg = LearningMADDPG()
    rclpy.spin(learning_maddpg)
    learning_maddpg.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
