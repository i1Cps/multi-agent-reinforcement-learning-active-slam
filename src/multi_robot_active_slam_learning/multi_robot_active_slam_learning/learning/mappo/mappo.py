from agent import Agent
from multi_robot_active_slam_learning.learning.mappo.agents import Agent


class MAPPO:
    def __init__(
        self,
        actor_dims,
        critic_dims,
        n_agents,
        n_actions,
        n_epochs,
        env,
        T,
        n_procs,
        alpha=1e-4,
        fc1=64,
        fc2=64,
        gamma=0.95,
        tau=0.01,
        checkpoint_dir="models/",
        scenario="co-op_navigation",
    ):
        self.agents = []
        checkpoint_dir += scenario
        for agent_idx, agent in enumerate(env.agents):
            self.agents.append(
                Agent(
                    actor_dims=actor_dims[agent],
                    critic_dims=critic_dims,
                    n_actions=n_actions[agent],
                    agent_idx=agent_idx,
                    alpha=alpha,
                    fc1=fc1,
                    fc2=fc2,
                    gamma=gamma,
                    checkpoint_dir=checkpoint_dir,
                    scenario=scenario,
                    name="",
                )
            )

    def choose_actions(self, raw_obs):
        actions = {}
        probs = {}
        for agent_id, agent in zip(raw_obs, self.agents):
            action, prob = agent.choose_action(raw_obs[agent_id])
            actions[agent_id] = action
            probs[agent_id] = prob
        return actions, probs

    def learn(self, memory):
        for agent in self.agents:
            agent.learn(memory, self.agents)

    def save_checkpoint(self):
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        for agent in self.agents:
            agent.load_models()
