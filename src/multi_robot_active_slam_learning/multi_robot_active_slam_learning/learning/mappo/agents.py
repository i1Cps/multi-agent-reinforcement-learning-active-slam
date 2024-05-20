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
        actor_dims: int,
        critic_dims: int,
        n_actions: int,
        agent_idx: int,
        gamma: float = 0.99,
        alpha: float = 3e-4,
        beta: float = 3e-4,
        actor_fc1: int = 128,
        actor_fc2: int = 128,
        critic_fc1: int = 128,
        critic_fc2: int = 128,
        entropy_coefficient: float = 1e-3,
        gae_lambda: float = 0.95,
        policy_clip: float = 0.2,
        n_epochs: int = 10,
        checkpoint_dir: str = "models/",
        scenario: str = "unclassified",
    ):
        self.gamma = gamma
        self.alpha = alpha

        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.entropy_coefficient = entropy_coefficient
        self.agent_idx = agent_idx
        self.actor = ActorNetwork(
            input_dims=actor_dims,
            n_actions=n_actions,
            learning_rate=alpha,
            fc1=actor_fc1,
            fc2=actor_fc2,
            checkpoint_dir=checkpoint_dir,
            scenario=scenario,
            name=f"agent_{agent_idx}_actor",
        )
        self.critic = CriticNetwork(
            input_dims=critic_dims,
            learning_rate=beta,
            fc1=critic_fc1,
            fc2=critic_fc2,
            checkpoint_dir=checkpoint_dir,
            scenario=scenario,
            name=f"agent_{agent_idx}_critic",
        )
        self.n_actions = n_actions

    def choose_action(self, observation: np.ndarray) -> tuple:
        with T.no_grad():
            state = T.tensor(observation, dtype=T.float, device=self.actor.device)
            dist = self.actor(state)
            action = dist.sample()
            probs = dist.log_prob(action)
        return action.cpu().numpy().flatten(), probs.cpu().numpy().flatten()

    def calc_adv_and_returns(self, memories: tuple) -> tuple:
        states, next_states, rewards, dones = memories
        with T.no_grad():
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()

            deltas = rewards[:, self.agent_idx] + self.gamma * next_values - values
            deltas = deltas.cpu().numpy()
            adv = [0]
            for step in reversed(range(deltas.shape[0])):
                done_step = np.array(dones[step].cpu())
                advantage = (
                    deltas[step] + self.gamma * self.gae_lambda * adv[-1] * done_step
                )
                adv.append(advantage)
            adv.reverse()
            adv = np.array(adv[:-1])
            adv = T.tensor(adv, device=self.critic.device).unsqueeze(1)
            returns = adv + values.unsqueeze(1)
            adv = (adv - adv.mean()) / (adv.std() + 1e-4)
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
        actions_array = T.tensor(actions[self.agent_idx], dtype=T.float, device=device)
        rewards_array = T.tensor(rewards, dtype=T.float, device=device)
        actor_states_array = T.tensor(
            actor_states[self.agent_idx], dtype=T.float, device=device
        )
        terminated_array = T.tensor(terminated, dtype=T.float, device=device)
        old_probs_array = T.tensor(
            old_probs[self.agent_idx], dtype=T.float, device=device
        )

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
                    new_probs.sum(1, keepdim=True) - old_probs.sum(1, keepdim=True)
                )

                weighted_probs = adv[batch] * prob_ratio
                weighted_clipped_probs = (
                    T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                    * adv[batch]
                )

                entropy = dist.entropy().sum(1, keepdim=True)
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs)
                actor_loss -= self.entropy_coefficient * entropy
                self.actor.optimizer.zero_grad()
                actor_loss.mean().backward()

                T.nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
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
