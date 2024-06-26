from multi_robot_active_slam_learning.learning.maddpg.agents import Agent
from multi_robot_active_slam_learning.learning.maddpg.memory import (
    MultiAgentReplayBuffer,
)
from typing import List, Dict, Any
import numpy as np


class MADDPG:
    def __init__(
        self,
        actor_input_dims: List[int],
        critic_input_dims: int,
        n_agents: int,
        n_actions: List[int],
        alpha: float,
        beta: float,
        gamma: float,
        tau: float,
        max_actions: np.ndarray,
        min_actions: np.ndarray,
        actor_fc1: int = 256,
        actor_fc2: int = 256,
        critic_fc1: int = 256,
        critic_fc2: int = 256,
    ):
        # Handle agent instances in a list
        self.agents = []
        self.n_actions = n_actions
        for agent_idx in range(n_agents):
            self.agents.append(
                Agent(
                    actor_dims=actor_input_dims[agent_idx],
                    critic_dims=critic_input_dims,
                    alpha=alpha,
                    beta=beta,
                    tau=tau,
                    gamma=gamma,
                    agent_idx=agent_idx,
                    n_actions=n_actions[agent_idx],
                    min_actions=min_actions,
                    max_actions=max_actions,
                    actor_fc1=actor_fc1,
                    actor_fc2=actor_fc2,
                    critic_fc1=critic_fc1,
                    critic_fc2=critic_fc2,
                )
            )

    # Choose actions for each agent
    def choose_actions(self, single_obs: List) -> np.ndarray:
        actions = np.empty((len(self.agents), self.n_actions[0]), dtype=np.float32)
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(single_obs[agent_idx], eval=False)
            actions[agent_idx] = action
        return actions

    # Choose random action for each agent
    def choose_random_actions(self) -> np.ndarray:
        random_actions = np.empty(
            (len(self.agents), self.n_actions[0]), dtype=np.float32
        )
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_random_actions()
            random_actions[agent_idx] = action
        return random_actions

    def learn(self, memory: MultiAgentReplayBuffer):
        for agent in self.agents:
            agent.learn(memory, self.agents)

    def reset_noise(self) -> None:
        for agent in self.agents:
            agent.ou_noise.reset()

    def save(self, filepath) -> None:
        for agent in self.agents:
            agent.save(filepath)

    def load(self, filepath) -> None:
        for agent in self.agents:
            agent.load(filepath)
