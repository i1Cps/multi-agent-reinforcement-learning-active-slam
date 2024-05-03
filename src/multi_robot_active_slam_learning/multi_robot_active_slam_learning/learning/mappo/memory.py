import numpy as np


class PPOMemory:
    def __init__(
        self,
        batch_size,
        T,
        n_agents,
        agents,
        n_procs,
        critic_dims,
        actor_dims,
        n_actions,
    ):
        self.states = np.zeros((T, n_procs, critic_dims), dtype=np.float32)
        self.rewards = np.zeros((T, n_procs, n_agents), dtype=np.float32)
        self.dones = np.zeros((T, n_procs), dtype=np.float32)
        self.next_states = np.zeros((T, n_procs, critic_dims), dtype=np.float32)

        self.actor_states = {a: np.zeros((T, n_procs, actor_dims[a])) for a in agents}
        self.actor_next_states = {
            a: np.zeros((T, n_procs, actor_dims[a])) for a in agents
        }
        self.actions = {a: np.zeros((T, n_procs, n_actions[a])) for a in agents}
        self.probs = {a: np.zeros((T, n_procs, n_actions[a])) for a in agents}

        self.memory_counter = 0
        self.n_states = T
        self.n_procs = n_procs
        self.critic_dims = critic_dims
        self.actor_dims = actor_dims
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.agents = agents
        self.batch_size = batch_size

    def recall(self):
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

    # Generate batches, but we will return the indices for the learn function to index.
    # Reason being is that the learn function will first call "recall"

    def generate_batches(self):
        n_states = len(self.states)
        n_batches = int(n_states // self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [
            indices[i * self.batch_size : (i + 1) * self.batch_size]
            for i in range(n_batches)
        ]
        return batches

    def store_memory(
        self, raw_obs, state, action, reward, next_raw_obs, next_state, terminal, prob
    ):
        index = self.memory_counter % self.n_states
        self.states[index] = state
        self.next_states[index] = next_state
        self.dones[index] = terminal
        self.rewards[index] = reward

        for agent in self.agents:
            self.actions[agent][index] = action[agent]
            self.actor_states[agent][index] = raw_obs[agent]
            self.actor_next_states[agent][index] = next_raw_obs[agent]
        self.memory_counter += 1

    def clear_memory(self):
        self.states = np.zeros(
            (self.n_states, self.n_procs, self.critic_dims), dtype=np.float32
        )
        self.rewards = np.zeros(
            (self.n_states, self.n_procs, self.n_agents), dtype=np.float32
        )
        self.dones = np.zeros((self.n_states, self.n_procs), dtype=np.float32)
        self.next_states = np.zeros(
            (self.n_states, self.n_procs, self.critic_dims), dtype=np.float32
        )

        self.actor_states = {
            a: np.zeros((self.n_states, self.n_procs, self.actor_dims[a]))
            for a in self.agents
        }
        self.actor_next_states = {
            a: np.zeros((self.n_states, self.n_procs, self.actor_dims[a]))
            for a in self.agents
        }
        self.actions = {
            a: np.zeros((self.n_states, self.n_procs, self.n_actions[a]))
            for a in self.agents
        }
        self.probs = {
            a: np.zeros((self.n_states, self.n_procs, self.n_actions[a]))
            for a in self.agents
        }
