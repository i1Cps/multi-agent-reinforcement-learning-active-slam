import numpy as np
import copy
from typing import List
import torch as T
import torch.nn.functional as F
from multi_robot_active_slam_learning.learning.maddpg.memory import (
    MultiAgentReplayBuffer,
)

from multi_robot_active_slam_learning.learning.maddpg.networks import (
    ActorNetwork,
    CriticNetwork,
)

from multi_robot_active_slam_learning.learning.maddpg.noise import (
    DifferentialDriveOUNoise,
)


class Agent:
    def __init__(
        self,
        actor_dims: int,
        critic_dims: int,
        n_actions: int,
        agent_idx: int,
        min_actions: np.ndarray,
        max_actions: np.ndarray,
        alpha: float,
        beta: float,
        tau: float,
        gamma: float = 0.95,
        actor_fc1: int = 256,
        actor_fc2: int = 256,
        critic_fc1: int = 256,
        critic_fc2: int = 256,
    ):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = "agent_%s" % agent_idx
        self.agent_idx = agent_idx
        self.min_actions = min_actions
        self.max_actions = max_actions

        self.actor = ActorNetwork(
            input_dims=actor_dims,
            learning_rate=alpha,
            fc1=actor_fc1,
            fc2=actor_fc2,
            n_actions=n_actions,
            max_actions=max_actions,
        )

        self.target_actor = ActorNetwork(
            input_dims=actor_dims,
            learning_rate=alpha,
            fc1=actor_fc1,
            fc2=actor_fc2,
            n_actions=n_actions,
            max_actions=max_actions,
        )

        self.critic = CriticNetwork(
            input_dims=critic_dims,
            learning_rate=beta,
            fc1=critic_fc1,
            fc2=critic_fc2,
        )

        self.target_critic = CriticNetwork(
            input_dims=critic_dims,
            learning_rate=beta,
            fc1=critic_fc1,
            fc2=critic_fc2,
        )

        self.ou_noise = DifferentialDriveOUNoise(
            mean=0,
            theta=0.15,
            sigma=0.2,
            dt=0.01,
            max_angular=max_actions[1],
            min_angular=min_actions[1],
            max_linear=max_actions[0],
            min_linear=min_actions[0],
        )

        # Set network and target network weights to equal each other
        self.update_network_parameters(tau=1)

    def choose_action(self, observation: np.ndarray, eval: bool = False) -> np.ndarray:
        with T.no_grad():
            # Convert observation to tensor
            state = T.tensor(
                observation[np.newaxis, :], dtype=T.float, device=self.actor.device
            )
            # Generate actions using the actor network
            mu = self.actor.forward(state).to(self.actor.device).cpu().numpy()[0]
            noise = np.random.normal(0, self.max_actions * 0.1, size=self.n_actions)
            # When evaluating we dont want noise
            noise *= 1 - int(eval)
            # Add noise and clip
            action = (mu + noise).clip(self.min_actions, self.max_actions)
            # Ensure the action is float32
            return action.astype(np.float32)

    def choose_random_actions(self):
        # Generate action using Ornstein–Uhlenbeck noise
        return self.ou_noise()

    def learn(self, memory: MultiAgentReplayBuffer, agent_list):
        # Check if enough memory is in the buffer before sampling
        if not memory.ready():
            return

        # Sample a batch of memories
        (
            actor_states,
            states,
            actions,
            rewards,
            next_actor_states,
            next_states,
            dones,
        ) = memory.sample_buffer()

        device = self.actor.device

        states = T.tensor(np.array(states), dtype=T.float, device=device)
        rewards = T.tensor(np.array(rewards), dtype=T.float, device=device)
        next_states = T.tensor(np.array(next_states), dtype=T.float, device=device)
        dones = T.tensor(np.array(dones), device=device)

        actor_states = [
            T.tensor(actor_states[idx], device=device, dtype=T.float)
            for idx in range(len(agent_list))
        ]
        next_actor_states = [
            T.tensor(next_actor_states[idx], device=device, dtype=T.float)
            for idx in range(len(agent_list))
        ]
        actions = [
            T.tensor(actions[idx], device=device, dtype=T.float)
            for idx in range(len(agent_list))
        ]

        # ------------------- Update Critic ------------------------------ #

        # long story short ..... read the paper
        with T.no_grad():
            next_actions = T.cat(
                [
                    agent.target_actor(next_actor_states[idx])
                    for idx, agent in enumerate(agent_list)
                ],
                dim=1,
            )
            Q_critic_next = self.target_critic.forward(
                next_states, next_actions
            ).squeeze()
            Q_critic_next[dones[:, self.agent_idx]] = 0.0

            Q_target = rewards[:, self.agent_idx] + self.gamma * Q_critic_next

        old_actions = T.cat([actions[idx] for idx in range(len(agent_list))], dim=1)
        Q_critic = self.critic.forward(states, old_actions).squeeze()

        # Loss Calculation
        critic_loss = F.mse_loss(Q_critic, Q_target)
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # ------------------------ Update Actor -------------------------------- #

        # The most hard to grasp part from the paper for me.

        # Critic network still critiques everyones actions in actor loss similiar to ddpg,
        # except we update the action for this agent with the current policy

        # Update THIS agent action
        actions[self.agent_idx] = self.actor.forward(actor_states[self.agent_idx])
        # Concatonate updated actions array
        actions = T.cat([actions[i] for i in range(len(agent_list))], dim=1)

        # Loss Calculation
        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, actions).mean()
        actor_loss.backward()
        self.actor.optimizer.step()

        # ------------------ Update Target Networks ---------------------------------- #

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        tau = tau or self.tau
        src = self.actor
        dest = self.target_actor

        for param, target in zip(src.parameters(), dest.parameters()):
            target.data.copy_(tau * param.data + (1 - tau) * target.data)

        src = self.critic
        dest = self.target_critic

        for param, target in zip(src.parameters(), dest.parameters()):
            target.data.copy_(tau * param.data + (1 - tau) * target.data)

    def save(self, filepath):
        T.save(
            self.actor.state_dict(), filepath / ("maddpg_actor_" + str(self.agent_idx))
        )
        T.save(
            self.actor.optimizer.state_dict(),
            filepath / ("maddpg_actor_optimiser_" + str(self.agent_idx)),
        )

        T.save(
            self.critic.state_dict(),
            filepath / ("maddpg_critic_" + str(self.agent_idx)),
        )
        T.save(
            self.critic.optimizer.state_dict(),
            filepath / ("maddpg_critic_optimiser_" + str(self.agent_idx)),
        )

        print("... saving checkpoint ...")

    def load(self, filepath):
        self.actor.load_state_dict(
            (T.load(filepath / ("maddpg_actor_" + str(self.agent_idx))))
        )
        self.actor.optimizer.load_state_dict(
            T.load(filepath / ("maddpg_actor_optimiser_" + str(self.agent_idx)))
        )
        self.target_actor = copy.deepcopy(self.actor)

        self.critic.load_state_dict(
            (T.load(filepath / ("maddpg_critic_" + str(self.agent_idx))))
        )
        self.critic.optimizer.load_state_dict(
            T.load(filepath / ("maddpg_critic_optimiser_" + str(self.agent_idx)))
        )
        self.target_critic = copy.deepcopy(self.critic)

        print("... loading checkpoint ...")
