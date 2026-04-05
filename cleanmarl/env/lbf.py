import numpy as np
from .common_interface import CommonInterface

import lbforaging
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium.spaces import Tuple, flatdim


class LBFWrapper(CommonInterface):
    def __init__(self, map_name, reward_aggr='sum', seed=0, time_limit=150, agent_ids=False, **kwargs):
        super().__init__()
        # FIX 1: Do NOT pass max_episode_steps to gym.make() — gymnasium already wraps
        # the env in TimeLimit when that arg is given, causing double-wrapping and
        # potentially infinite episodes when the inner wrapper's counter resets first.
        self.env = gym.make(map_name, **kwargs)
        self.env = TimeLimit(self.env, max_episode_steps=time_limit)
        self.agent_ids = agent_ids
        self.n_agents = self.env.unwrapped.n_agents
        self.agents = list(range(self.n_agents))
        self.episode_limit = time_limit
        self.reward_aggr = reward_aggr
        self.action_space = Tuple(
            tuple([self.env.action_space[agent] for agent in self.agents]))
        self.longest_action_space = max(self.env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(self.env.observation_space, key=lambda x: x.shape)
        # FIX 2: Cache identity matrix — avoids allocating a new (n_agents × n_agents)
        # float64 array on every single environment step.
        self._agent_id_matrix = np.eye(self.n_agents, dtype=np.float32)
        # Initialise per-episode step counter here so reset() is not required before step().
        self.current_step = 0
        self.state = None
        self.last_reward_vector = np.zeros(self.n_agents, dtype=np.float32)

    def step(self, actions):
        """Returns obs, reward, terminated, truncated, info."""
        actions = [int(act) for act in actions]
        obs, reward, terminated, truncated, info = self.env.step(actions)
        self.current_step += 1

        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, "_food_spawned") and hasattr(unwrapped, "field"):
            food_spawned = unwrapped._food_spawned
            food_remaining = np.count_nonzero(unwrapped.field)
            info["food_collected_ratio"] = (food_spawned - food_remaining) / food_spawned if food_spawned > 0 else 0.0


        # Store per-agent rewards before aggregating so NashConv helpers can use them.
        if isinstance(reward, (list, tuple, np.ndarray)):
            self.last_reward_vector = np.asarray(reward, dtype=np.float32)
        else:
            self.last_reward_vector = np.full(self.n_agents, float(reward), dtype=np.float32)

        obs = self.process_obs(obs)

        if self.reward_aggr == "sum":
            scalar_reward = np.float32(np.sum(self.last_reward_vector))
        elif self.reward_aggr == "mean":
            scalar_reward = np.float32(np.mean(self.last_reward_vector))
        else:
            scalar_reward = np.float32(self.last_reward_vector[0])

        # FIX 3: Remove the broken manual truncation check.
        # TimeLimit already sets truncated=True at the episode limit; the old check
        # `if terminated and self.current_step == self.env.unwrapped._max_episode_steps`
        # was (a) logically wrong — truncation is signalled via truncated, not terminated —
        # and (b) would raise AttributeError because _max_episode_steps lives on the
        # TimeLimit wrapper, not on the unwrapped env.

        return obs, scalar_reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        self.current_step = 0
        self.last_reward_vector = np.zeros(self.n_agents, dtype=np.float32)
        obs, _ = self.env.reset(seed=seed, options=options)
        obs = self.process_obs(obs)
        return obs, {}

    def get_obs_size(self):
        """Returns the shape of the observation (per agent)."""
        return flatdim(self.longest_observation_space) + self.agent_ids * self.n_agents

    def get_state_size(self):
        """Returns the size of the global state."""
        return flatdim(self.longest_observation_space) * self.n_agents

    def get_state(self):
        """Returns the global state (concatenation of all agent observations)."""
        return self.state

    def get_action_size(self):
        return self.longest_action_space.n

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_actions.append(self.get_avail_agent_actions(agent_id))
        return np.array(avail_actions)

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        valid = flatdim(self.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid

    def get_last_reward_vector(self):
        """Returns the per-agent reward from the last step."""
        return self.last_reward_vector.copy()

    def sample(self):
        return list(self.env.action_space.sample())

    def process_obs(self, obs):
        obs = np.array(obs, dtype=np.float32)
        # FIX 4: Use .copy() so self.state is a standalone array, not a view into obs.
        # Without this, both the returned obs and self.state share the same underlying
        # memory buffer; if obs is later modified in-place the state would be corrupted,
        # and the ref-count on that buffer is never dropped until *both* variables are freed.
        self.state = obs.reshape(-1).copy()
        if self.agent_ids:
            # Use the cached identity matrix instead of re-allocating every step.
            obs = np.concatenate((obs, self._agent_id_matrix), axis=1)
        return obs

    def close(self):
        return self.env.close()