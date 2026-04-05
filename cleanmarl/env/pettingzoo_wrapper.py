from .common_interface import CommonInterface
from gymnasium.spaces import Box, flatdim
import importlib
import numpy as np


class PettingZooWrapper(CommonInterface):
    def __init__(self, family, env_name, agent_ids=False, **kwargs):
        env = importlib.import_module(f'pettingzoo.{family}.{env_name}')
        self.env = env.parallel_env(**kwargs)
        self.env.reset()
        self.n_agents = self.env.num_agents
        self.agents = self.env.agents
        self.act_dim = flatdim(self.env.action_space(self.agents[0]))
        if isinstance(self.env.action_space(self.agents[0]), Box):
            self.act_low = self.env.action_space(self.agents[0]).low
            self.act_high = self.env.action_space(self.agents[0]).high
        self.obs_dims = {
            agent: flatdim(self.env.observation_space(agent)) for agent in self.agents
        }
        self.obs_dim = max(self.obs_dims.values())
        self.agent_ids = agent_ids
        self.last_reward_vector = np.zeros(self.n_agents, dtype=np.float32)

    def reset(self, seed=None):
        obs, _ = self.env.reset(seed=seed)
        obs = self.process_obs(obs)
        self.last_obs = obs
        self.last_reward_vector = np.zeros(self.n_agents, dtype=np.float32)
        return obs, {}

    def render(self, mode='human'):
        return self.env.render(mode)

    def step(self, actions):
        dict_actions = {agent: actions[index] for index, agent in enumerate(self.agents)}
        observations, rewards, dones, truncated, infos = self.env.step(dict_actions)

        rewards = np.asarray([rewards.get(agent, 0.0) for agent in self.agents], dtype=np.float32)
        self.last_reward_vector = rewards.copy()
        done_flags = [dones.get(agent, False) for agent in self.agents]
        truncated_flags = [truncated.get(agent, False) for agent in self.agents]
        done = all(done_flags)
        truncated = all(truncated_flags)
        has_observations = observations is not None and len(observations) != 0
        if not has_observations and not any(done_flags) and not any(truncated_flags):
            done = True
        info = {f'{agent}_{key}': value for agent in self.agents for key, value in infos.get(agent, {}).items()}
        info["reward_vector"] = self.last_reward_vector.copy()

        if has_observations:
            obs = self.process_obs(observations)
            self.last_obs = obs
        else:
            obs = self.last_obs

        return obs, float(rewards[0]), done, truncated, info

    def get_obs_size(self):
        return self.obs_dim + self.agent_ids * self.n_agents

    def get_state_size(self):
        return self.obs_dim * self.n_agents

    def get_state(self):
        return self.state

    def get_action_size(self):
        return self.act_dim

    def get_last_reward_vector(self):
        return self.last_reward_vector.copy()

    def get_avail_actions(self):
        return np.ones((self.n_agents, self.act_dim))

    def sample(self):
        return [self.env.action_space(agent).sample() for agent in self.agents]

    def process_obs(self, obs):
        if obs is None:
            obs = {}
        padded_obs = []
        for agent in self.agents:
            if agent in obs:
                agent_obs = np.asarray(obs[agent], dtype=np.float32).reshape(-1)
            else:
                agent_obs = np.zeros(self.obs_dims[agent], dtype=np.float32)
            if agent_obs.size < self.obs_dim:
                agent_obs = np.pad(agent_obs, (0, self.obs_dim - agent_obs.size))
            padded_obs.append(agent_obs)
        obs = np.stack(padded_obs, axis=0)
        self.state = obs.reshape(-1)
        if self.agent_ids:
            obs = np.concatenate((obs, np.eye(self.n_agents, dtype=obs.dtype)), axis=1)
        return obs

    def close(self):
        return self.env.close()
