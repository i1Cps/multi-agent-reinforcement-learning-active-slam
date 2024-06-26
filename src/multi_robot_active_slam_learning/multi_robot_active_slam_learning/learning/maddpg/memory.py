import numpy as np
from typing import List, Tuple


class MultiAgentReplayBuffer:
    def __init__(
        self,
        max_size: int,
        critic_state_dims: int,
        actor_state_dims: List[int],
        n_actions: List[int],
        n_agents: int,
        batch_size: int,
    ):
        self.mem_size = max_size
        self.mem_counter = 0
        self.n_agents = n_agents
        self.actor_state_dims = actor_state_dims
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.state_memory = np.zeros((self.mem_size, critic_state_dims))
        self.next_state_memory = np.zeros((self.mem_size, critic_state_dims))
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)

        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_next_state_memory = []
        self.actor_action_memory = []

        for i in range(self.n_agents):
            self.actor_state_memory.append(
                np.zeros((self.mem_size, self.actor_state_dims[i]))
            )
            self.actor_next_state_memory.append(
                np.zeros((self.mem_size, self.actor_state_dims[i]))
            )
            self.actor_action_memory.append(
                np.zeros((self.mem_size, self.n_actions[i]))
            )

    def store_transition(
        self,
        single_obs: List[np.ndarray],
        global_obs: np.ndarray,
        actions: List[np.ndarray],
        rewards: List,
        next_single_obs: List[np.ndarray],
        next_global_obs: np.ndarray,
        dones: List[bool],
    ):
        index = self.mem_counter % self.mem_size
        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = single_obs[agent_idx]
            self.actor_next_state_memory[agent_idx][index] = next_single_obs[agent_idx]
            self.actor_action_memory[agent_idx][index] = actions[agent_idx]

        self.state_memory[index] = global_obs
        self.next_state_memory[index] = next_global_obs
        self.reward_memory[index] = rewards
        self.terminal_memory[index] = dones
        self.mem_counter += 1

    def sample_buffer(
        self,
    ) -> Tuple[
        List[np.ndarray],
        np.ndarray,
        List[np.ndarray],
        np.ndarray,
        List[np.ndarray],
        np.ndarray,
        np.ndarray,
    ]:
        max_mem = min(self.mem_counter, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        states = self.state_memory[batch]
        next_states = self.next_state_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        actor_states = []
        actor_next_states = []
        actions = []

        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_next_states.append(self.actor_next_state_memory[agent_idx][batch])
            actions.append(self.actor_action_memory[agent_idx][batch])

        return (
            actor_states,
            states,
            actions,
            rewards,
            actor_next_states,
            next_states,
            terminal,
        )

    def ready(self) -> bool:
        return self.mem_counter >= self.batch_size
