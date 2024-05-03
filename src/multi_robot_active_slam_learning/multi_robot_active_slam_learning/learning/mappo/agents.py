import numpy as np
import torch as T
import torch.nn.functional as F
from multi_robot_active_slam_learning.learning.mappo.networks import (
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
        name,
        gamma=0.99,
        alpha=3e-4,
        fc1=2,
        fc2=4,
        entrophy_coefficient=1e-3,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=64,
        n_epochs=10,
        n_procs=8,
        checkpoint_dir=None,
        scenario=None,
    ):
        self.gamma = gamma
        self.alpha = alpha

        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.entrophy_coefficient = entrophy_coefficient
        self.agent_idx = agent_idx
        self.name = name
        self.n_procs = n_procs
        self.actor = ContinuousActorNetwork(
            input_dims=actor_dims,
            n_actions=n_actions,
            alpha=alpha,
            checkpoint_dir=checkpoint_dir,
            scenario=scenario,
        )
        self.critic = ContinuousCriticNetwork(
            input_dims=critic_dims, alpha=alpha, scenario=scenario
        )
        self.n_actions = n_actions

    def choose_action(self, observation):
        with T.no_grad():
            state = T.tensor(observation, dtype=T.float, device=self.actor.device)

            dist = self.actor(state)
            action = dist.sample()
            probs = dist.log_prob(action)

        return action.cpu().numpy().flatten(), probs.cpu().numpy().flatten()

    def calc_adv_and_returns(self, memories):
        states, next_states, rewards, terminated = memories
        with T.no_grad():
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
            deltas = rewards[:, :, self.agent_idx] + self.gamma * next_values - values
            deltas = deltas.cpu().numpy()
            adv = [0]
            for step in reversed(range(deltas.shape[0])):
                advantage = deltas[step] + self.gamma * self.gae_lambda * adv[
                    -1  # kmt
                ] * np.array(terminated[step])
                adv.append(advantage)
            adv.reverse()
            adv = np.array(adv[:-1])
            adv = T.tensor(adv, device=self.critic.device).unsqueeze(2)
            returns = adv + values.unsqueeze(2)
            adv = (adv - adv.mean()) / (adv.std() + 1e-4)  # Normalisation per say
        return adv, returns

    def learn(self, memory):
        (
            actor_states,
            states,
            actions,
            old_probs,
            rewards,
            actor_next_states,
            next_states,
            terminated,
        ) = memory.recall()

        device = self.critic.device
        states_array = T.tensor(states, dtype=T.float, device=device)
        next_states_array = T.tensor(next_states, dtype=T.float, device=device)
        actions_array = T.tensor(actions, dtype=T.float, device=device)
        rewards_array = T.tensor(rewards, dtype=T.float, device=device)
        actor_states_array = T.tensor(actor_states, dtype=T.float, device=device)
        terminated_array = T.tensor(terminated, dtype=T.float, device=device)
        old_probs_array = T.tensor(old_probs, dtype=T.float, device=device)

        adv, returns = self.calc_adv_and_returns(
            (states_array, next_states_array, rewards_array, terminated_array)
        )

        for epoch in range(self.n_epochs):
            batches = memory.generate_batches()
            for batch in batches:
                old_probs = old_probs_array[batch]
                actions = actions_array[batch]
                actor_states = actor_states_array[batch]

                dist = self.actor(actor_states)
                new_probs = dist.log_prob(actions)
                prob_ratio = T.exp(
                    new_probs.sum(2, keepdim=True) - old_probs.sum(2, keepdim=True)
                )
                weighted_probs = adv[batch] * prob_ratio
                weighted_clipped_probs = (
                    T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                    * adv[batch]
                )

                entropy = dist.entropy().sum(2, keepdim=True)
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs)
                actor_loss -= self.entrophy_coefficient * entropy
                self.actor.optimizer.zero_grad()
                actor_loss.mean().backward()

                T.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor.optimizer.step()

                states = states_array[batch]
                critic_value = self.critic(states).squeeze()
                critic_loss = (critic_value - returns[batch].squeeze()).pow(2).mean()
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()

    def save_models(self):
        self.critic.save_checkpoint()
        self.actor.save_checkpoint()

    def load_models(self):
        self.critic.load_checkpoint()
        self.actor.load_checkpoint()
