from multi_robot_active_slam_learning.learning.mappo.agents import Agent
from multi_robot_active_slam_learning.learning.mappo.memory import MAPPOMemory
import numpy as np
from typing import List


class MAPPO:
    def __init__(
        self,
        actor_dims: List[int],
        critic_dims: int,
        n_agents: int,
        n_actions: List[int],
        entropy_coefficient: float = 1e-3,
        gae_lambda: float = 0.95,
        policy_clip: float = 0.2,
        n_epochs: int = 10,
        alpha: float = 1e-4,
        beta: float = 1e-4,
        actor_fc1: int = 64,
        actor_fc2: int = 64,
        critic_fc1: int = 64,
        critic_fc2: int = 64,
        gamma: float = 0.99,
        checkpoint_dir: str = "models/",
        scenario: str = "unclassified",
    ):
        self.agents = []
        self.n_actions = n_actions
        for agent_idx in range(n_agents):
            self.agents.append(
                Agent(
                    actor_dims=actor_dims[agent_idx],
                    critic_dims=critic_dims,
                    n_actions=n_actions[agent_idx],
                    agent_idx=agent_idx,
                    alpha=alpha,
                    beta=beta,
                    n_epochs=n_epochs,
                    entropy_coefficient=entropy_coefficient,
                    gae_lambda=gae_lambda,
                    policy_clip=policy_clip,
                    actor_fc1=actor_fc1,
                    actor_fc2=actor_fc2,
                    critic_fc1=critic_fc1,
                    critic_fc2=critic_fc2,
                    gamma=gamma,
                    checkpoint_dir=checkpoint_dir,
                    scenario=scenario,
                )
            )

    def choose_actions(self, raw_obs: List) -> tuple[np.ndarray, np.ndarray]:
        actions = np.empty((len(self.agents), self.n_actions[0]), dtype=np.float32)
        probs = np.empty((len(self.agents), self.n_actions[0]), dtype=np.float32)
        for agent_idx, agent in enumerate(self.agents):
            action, prob = agent.choose_action(raw_obs[agent_idx])
            actions[agent_idx] = action
            probs[agent_idx] = prob
        return actions, probs

    def learn(self, memory: MAPPOMemory):
        for agent in self.agents:
            agent.learn(memory)

    def save_checkpoint(self):
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        for agent in self.agents:
            agent.load_models()
