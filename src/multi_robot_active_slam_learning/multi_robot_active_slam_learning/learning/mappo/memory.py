import numpy as np
from typing import List, Tuple


class MAPPOMemory:
    def __init__(
        self,
        batch_size: int,
        T: int,
        n_agents: int,
        critic_dims: int,
        actor_dims: List[int],
        n_actions: List[int],
    ):
        self.states = np.zeros((T, critic_dims), dtype=np.float32)
        self.rewards = np.zeros((T, n_agents), dtype=np.float32)
        self.dones = np.zeros((T), dtype=np.float32)
        self.next_states = np.zeros((T, critic_dims), dtype=np.float32)

        self.actor_states = []
        self.actor_next_states = []
        self.actions = []
        self.probs = []

        # Use dictionaries instead if you have heterogeneous agents
        for i in range(n_agents):
            self.actor_states.append(np.zeros((T, actor_dims[i])))
            self.actor_next_states.append(np.zeros((T, actor_dims[i])))
            self.actions.append(np.zeros((T, n_actions[i])))
            self.probs.append(np.zeros((T, n_actions[i])))

        self.memory_counter = 0
        self.n_states = T
        self.critic_dims = critic_dims
        self.actor_dims = actor_dims
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.batch_size = batch_size

    def recall(
        self,
    ) -> Tuple[
        List[np.ndarray],
        np.ndarray,
        List[np.ndarray],
        List[np.ndarray],
        np.ndarray,
        List[np.ndarray],
        np.ndarray,
        np.ndarray,
    ]:
        return (
            self.actor_states,
            self.states,
            self.actions,
            self.probs,
            self.rewards,
            self.actor_next_states,
            self.next_states,
            self.dones,
        )

    def generate_batches(self) -> List[np.ndarray]:
        n_batches = int(self.n_states // self.batch_size)
        indices = np.arange(self.n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [
            indices[i * self.batch_size : (i + 1) * self.batch_size]
            for i in range(n_batches)
        ]
        return batches

    def store_memory(
        self,
        raw_obs: List[np.ndarray],
        state: np.ndarray,
        action: np.ndarray,
        reward: List,
        next_raw_obs: List[np.ndarray],
        next_state: np.ndarray,
        terminal: bool,
        prob: np.ndarray,
    ):
        index = self.memory_counter % self.n_states
        self.states[index] = state
        self.next_states[index] = next_state
        self.dones[index] = terminal
        self.rewards[index] = reward

        for agent_idx in range(self.n_agents):
            self.actions[agent_idx][index] = action[agent_idx]
            self.probs[agent_idx][index] = prob[agent_idx]
            self.actor_states[agent_idx][index] = raw_obs[agent_idx]
            self.actor_next_states[agent_idx][index] = next_raw_obs[agent_idx]
        self.memory_counter += 1

    def clear_memory(self):
        self.states = np.zeros((self.n_states, self.critic_dims), dtype=np.float32)
        self.rewards = np.zeros((self.n_states, self.n_agents), dtype=np.float32)
        self.dones = np.zeros((self.n_states), dtype=np.float32)
        self.next_states = np.zeros((self.n_states, self.critic_dims), dtype=np.float32)

        self.actor_states = [
            np.zeros((self.n_states, self.actor_dims[a])) for a in range(self.n_agents)
        ]
        self.actor_next_states = [
            np.zeros((self.n_states, self.actor_dims[a])) for a in range(self.n_agents)
        ]
        self.actions = [
            np.zeros((self.n_states, self.n_actions[a])) for a in range(self.n_agents)
        ]
        self.probs = [
            np.zeros((self.n_states, self.n_actions[a])) for a in range(self.n_agents)
        ]
