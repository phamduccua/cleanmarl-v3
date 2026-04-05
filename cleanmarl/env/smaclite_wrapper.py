import torch
import numpy as np
from torch.distributions import Categorical

from .common_interface import CommonInterface

import gymnasium as gym
from gymnasium.spaces import flatdim
from gymnasium.wrappers import TimeLimit
import smaclite


class SMACliteWrapper(CommonInterface):
    def __init__(self, map_name, seed=0, time_limit=150, agent_ids=False, **kwargs):
        kwargs.setdefault("shared_reward", False)
        self.env = gym.make(f"smaclite/{map_name}-v0", seed=seed, **kwargs)
        self.env = TimeLimit(self.env, max_episode_steps=time_limit)
        self.agent_ids = agent_ids
        self.n_agents = self.env.unwrapped.n_agents
        self.episode_limit = time_limit
        self.longest_action_space = max(self.env.action_space, key=lambda x: x.n)
        self.agents = list(self.env.unwrapped.agents)
        self.last_reward_vector = np.zeros(self.n_agents, dtype=np.float32)
        self.last_obs = None

    def step(self, actions):
        actions = [int(act) for act in actions]
        obs, reward, terminated, truncated, info = self.env.step(actions)
        self.last_reward_vector = self._extract_reward_vector(reward, info)
        info = dict(info)
        info["battle_won"] = self._extract_battle_won(info)
        info["reward_vector"] = self.last_reward_vector.copy()
        obs = self.process_obs(obs)
        self.last_obs = obs
        return obs, reward, terminated, truncated, info

    def get_obs_size(self):
        return self.env.unwrapped.obs_size + self.agent_ids * self.n_agents

    def get_state_size(self):
        return self.env.unwrapped.state_size

    def get_state(self):
        return self.env.unwrapped.get_state()

    def get_action_size(self):
        return flatdim(self.longest_action_space)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = self.process_obs(obs)
        self.last_obs = obs
        self.last_reward_vector = np.zeros(self.n_agents, dtype=np.float32)
        info = dict(info)
        info["battle_won"] = 0.0
        info["reward_vector"] = self.last_reward_vector.copy()
        return obs, info

    def get_last_reward_vector(self):
        return self.last_reward_vector.copy()

    def get_avail_actions(self):
        return np.array(self.env.unwrapped.get_avail_actions())

    def get_agents(self):
        return self.agents

    def sample(self):
        avail_actions = torch.tensor(self.get_avail_actions(), dtype=torch.float32)
        masked_probs = avail_actions / avail_actions.sum(dim=1, keepdim=True)
        dist = Categorical(masked_probs)
        actions = dist.sample()
        return actions

    def process_obs(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if self.agent_ids:
            obs = np.concatenate((obs, np.eye(self.n_agents, dtype=obs.dtype)), axis=1)
        return obs

    def _extract_reward_vector(self, reward, info):
        if isinstance(info, dict):
            for key in ("reward_vector", "rewards", "per_agent_reward", "per_agent_rewards", "individual_rewards"):
                reward_vector = self._coerce_reward_vector(info.get(key))
                if reward_vector is not None:
                    return reward_vector
        for attr in ("last_reward_vector", "reward_vector", "rewards"):
            reward_vector = self._coerce_reward_vector(getattr(self.env.unwrapped, attr, None))
            if reward_vector is not None:
                return reward_vector
        reward_vector = self._coerce_reward_vector(reward)
        if reward_vector is not None:
            return reward_vector
        return np.zeros(self.n_agents, dtype=np.float32)

    def _coerce_reward_vector(self, value):
        if value is None:
            return None
        if isinstance(value, dict):
            if all(agent in value for agent in self.agents):
                return np.asarray([value[agent] for agent in self.agents], dtype=np.float32)
            return None
        reward_array = np.asarray(value, dtype=np.float32)
        if reward_array.ndim == 0:
            return np.full(self.n_agents, float(reward_array), dtype=np.float32)
        reward_array = reward_array.reshape(-1)
        if reward_array.size == self.n_agents:
            return reward_array.astype(np.float32)
        if reward_array.size == 1:
            return np.full(self.n_agents, float(reward_array[0]), dtype=np.float32)
        return None

    def _extract_battle_won(self, info):
        if isinstance(info, dict):
            for key in ("battle_won", "won", "win", "victory"):
                battle_won = self._coerce_battle_won(info.get(key))
                if battle_won is not None:
                    return battle_won
        for attr in ("battle_won", "won", "win", "victory"):
            battle_won = self._coerce_battle_won(getattr(self.env.unwrapped, attr, None))
            if battle_won is not None:
                return battle_won
        return 0.0

    def _coerce_battle_won(self, value):
        if value is None or isinstance(value, dict):
            return None
        battle_won = np.asarray(value, dtype=np.float32)
        if battle_won.ndim == 0:
            return float(battle_won)
        battle_won = battle_won.reshape(-1)
        if battle_won.size == 1:
            return float(battle_won[0])
        return float(battle_won.mean())

    def close(self):
        self.env.close()
