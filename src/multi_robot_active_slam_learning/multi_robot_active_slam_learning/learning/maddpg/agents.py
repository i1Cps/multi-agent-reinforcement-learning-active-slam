import numpy as np
import torch as T
import torch.nn.functional as F
from multi_robot_active_slam_learning.learning.maddpg.networks import (
    ActorNetwork,
    CriticNetwork,
)


class Agent:
    def __init__(
        self,
        actor_dims,
        critic_dims,
        n_actions,
        agent_idx,
        min_action,
        max_action,
        alpha=1e-4,
        beta=1e-3,
        actor_dims_fc1=256,
        actor_dims_fc2=256,
        critic_dims_fc1=256,
        critic_dims_fc2=256,
        gamma=0.99,
        tau=0.01,
        stacked_frames=1,
        checkpoint_dir="models",
    ):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = "agent_%s" % agent_idx
        self.agent_idx = agent_idx
        self.min_action = min_action
        self.max_action = max_action

        stacked_actor_dims = actor_dims * stacked_frames

        self.actor = ActorNetwork(
            alpha=alpha,
            input_dims=stacked_actor_dims,
            n_actions=n_actions,
            actor_dims_fc1=actor_dims_fc1,
            actor_dims_fc2=actor_dims_fc2,
            name=self.agent_name + "_actor",
            checkpoint_dir=checkpoint_dir,
        )

        self.target_actor = ActorNetwork(
            alpha=alpha,
            input_dims=stacked_actor_dims,
            n_actions=n_actions,
            actor_dims_fc1=actor_dims_fc1,
            actor_dims_fc2=actor_dims_fc2,
            name=self.agent_name + "_target_actor",
            checkpoint_dir=checkpoint_dir,
        )

        self.critic = CriticNetwork(
            beta=beta,
            input_dims=critic_dims,
            critic_dims_fc1=critic_dims_fc1,
            critic_dims_fc2=critic_dims_fc2,
            name=self.agent_name + "_critic",
            checkpoint_dir=checkpoint_dir,
        )

        self.target_critic = CriticNetwork(
            beta=beta,
            input_dims=critic_dims,
            critic_dims_fc1=critic_dims_fc1,
            critic_dims_fc2=critic_dims_fc2,
            name=self.agent_name + "_target_critic",
            checkpoint_dir=checkpoint_dir,
        )

        self.update_network_parameters(tau=1)

    def choose_action(self, observation, evaluate=False):
        with T.no_grad():  # Ensures that PyTorch will not track gradients
            state = T.tensor(
                observation[np.newaxis, :], dtype=T.float, device=self.actor.device
            )
            mu = self.actor.forward(state)  # Keep the output on the GPU

            # Generate noise using numpy and convert it to a PyTorch tensor on the appropriate device
            noise = np.random.normal(0, self.max_action * 0.1, self.n_actions)
            noise_tensor = T.tensor(noise, device=self.actor.device, dtype=T.float)

            action = T.clamp(
                mu + noise_tensor,
                T.tensor(-self.max_action, device=self.actor.device),
                T.tensor(self.max_action, device=self.actor.device),
            )
        # Convert to numpy array before returning, after all computations are complete
        return action.cpu().numpy()[0]

    def learn(self, memory, agent_list):
        if not memory.ready():
            return

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
        T.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic.optimizer.step()

        # ------------------------ Update Actor -------------------------------- #

        # The most hard to grasp part from the paper for me.

        # Critic network still critiques everyones actions in actor loss similiar to ddpg,
        # except we update the action for this agent

        actions[self.agent_idx] = self.actor.forward(actor_states[self.agent_idx])
        actions = T.cat([actions[i] for i in range(len(agent_list))], dim=1)
        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, actions).mean()
        actor_loss.backward()
        # T.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
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

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
