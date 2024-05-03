from multi_robot_active_slam_learning.learning.maddpg.agents import Agent
import numpy as np


class MADDPG:
    def __init__(
        self,
        actor_dims,
        critic_dims,
        n_agents,
        n_actions,
        max_action,
        min_action,
        alpha=1e-4,
        beta=1e-3,
        actor_dims_fc1=256,
        actor_dims_fc2=256,
        critic_dims_fc1=256,
        critic_dims_fc2=256,
        gamma=0.99,
        tau=0.005,
        stacked_frames=1,
        checkpoint_dir="",
        scenario="",
    ):
        self.agents = []
        self.n_actions = n_actions
        self.max_action = max_action
        self.min_action = min_action
        checkpoint_dir += scenario
        for agent_idx in range(n_agents):
            min_action = min_action
            max_action = max_action
            self.agents.append(
                Agent(
                    actor_dims=actor_dims[agent_idx],
                    critic_dims=critic_dims,
                    n_actions=n_actions[agent_idx],
                    agent_idx=agent_idx,
                    alpha=alpha,
                    tau=tau,
                    beta=beta,
                    actor_dims_fc1=actor_dims_fc1,
                    actor_dims_fc2=actor_dims_fc2,
                    critic_dims_fc1=critic_dims_fc1,
                    critic_dims_fc2=critic_dims_fc2,
                    stacked_frames=stacked_frames,
                    gamma=gamma,
                    min_action=min_action,
                    max_action=max_action,
                    checkpoint_dir=checkpoint_dir,
                )
            )

    def choose_actions(self, raw_obs, evaluate=False):
        actions = np.empty((len(self.agents), self.n_actions[0]), dtype=np.float32)
        for idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[idx], evaluate)
            actions[idx] = action
        return actions

    def choose_random_actions(self):
        # Initialize an array to hold the actions for all agents
        actions = np.array(
            [
                np.random.normal(0, agent.max_action * 0.2, size=agent.n_actions)
                for agent in self.agents
            ]
        )
        for idx, agent in enumerate(self.agents):
            actions[idx] = np.clip(actions[idx], -agent.max_action, agent.max_action)
        return actions

    def learn(self, memory):
        for agent in self.agents:
            agent.learn(memory, self.agents)

    def save_checkpoint(self):
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        for agent in self.agents:
            agent.load_models()
