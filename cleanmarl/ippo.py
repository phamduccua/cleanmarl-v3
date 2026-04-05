import torch
import tyro
import datetime
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import gc


@dataclass
class Args:
    env_type: str = "smaclite"
    """ Pettingzoo, SMAClite ... """
    env_name: str = "3m"
    """ Name of the environment"""
    env_family: str = "mpe"
    """ Env family when using pz"""
    agent_ids: bool = True
    """ Include id (one-hot vector) at the agent of the observations"""
    batch_size: int = 3
    """ Number of episodes to collect in each rollout"""
    actor_hidden_dim: int = 32
    """ Hidden dimension of actor network"""
    actor_num_layers: int = 1
    """ Number of hidden layers of actor network"""
    critic_hidden_dim: int = 32
    """ Hidden dimension of critic network"""
    critic_num_layers: int = 1
    """ Number of hidden layers of critic network"""
    optimizer: str = "Adam"
    """ The optimizer"""
    learning_rate_actor: float = 0.0008
    """ Learning rate for the actor"""
    learning_rate_critic: float = 0.0008
    """ Learning rate for the critic"""
    total_timesteps: int = 1000000
    """ Total steps in the environment during training"""
    gamma: float = 0.99
    """ Discount factor"""
    td_lambda: float = 0.95
    """ TD(λ) discount factor"""
    normalize_reward: bool = False
    """ Normalize the rewards if True"""
    normalize_advantage: bool = False
    """ Normalize the advantage if True"""
    normalize_return: bool = False
    """ Normalize the returns if True"""
    ppo_clip: float = 0.2
    """ PPO clipping factor """
    entropy_coef: float = 0.001
    """ Entropy coefficient """
    epochs: int = 3
    """ Number of training epochs"""
    clip_gradients: float = -1
    """ 0< for no clipping and 0> if clipping at clip_gradients"""
    log_every: int = 10
    """ Logging steps """
    eval_steps: int = 50
    """ Evaluate the policy each «eval_steps» training steps"""
    num_eval_ep: int = 10
    """ Number of evaluation episodes"""
    use_wnb: bool = False
    """ Logging to Weights & Biases if True"""
    wnb_project: str = ""
    """ Weights & Biases project name"""
    wnb_entity: str = ""
    """ Weights & Biases entity name"""
    device: str = "cpu"
    """ Device (cpu, cuda, mps)"""
    seed: int = 1
    """ Random seed"""
    approx_nashconv: bool = False
    """ Log an approximate NashConv metric during evaluation if True"""
    approx_nashconv_br_updates: int = 5
    """ Number of temporary best-response updates per agent"""
    approx_nashconv_br_episodes: int = 4
    """ Episodes collected for each temporary best-response update"""
    approx_nashconv_eval_episodes: int = 5
    """ Evaluation episodes used for each approximate Nash gap"""
    approx_nashconv_learning_rate: float = 0.0003
    """ Learning rate for the temporary best-response actor"""
    approx_nashconv_deterministic_eval: bool = True
    """ Use greedy actions when evaluating approximate Nash gaps"""


class RolloutBuffer:
    def __init__(
        self,
        buffer_size,
        num_agents,
        obs_space,
        state_space,
        action_space,
        normalize_reward=False,
        device="cpu",
    ):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_space = obs_space
        self.state_space = state_space
        self.action_space = action_space
        self.normalize_reward = normalize_reward
        self.device = device
        self.episodes = [None] * buffer_size
        self.pos = 0

    def add(self, episode):
        for key, values in episode.items():
            episode[key] = torch.from_numpy(np.stack(values)).float().to(self.device)
        self.episodes[self.pos] = episode
        self.pos += 1

    def get_batch(self):
        self.pos = 0
        lengths = [len(episode["obs"]) for episode in self.episodes]
        max_length = max(lengths)
        obs = torch.zeros(
            (self.buffer_size, max_length, self.num_agents, self.obs_space)
        ).to(self.device)
        avail_actions = torch.zeros(
            (self.buffer_size, max_length, self.num_agents, self.action_space)
        ).to(self.device)
        actions = torch.zeros((self.buffer_size, max_length, self.num_agents)).to(
            self.device
        )
        log_probs = torch.zeros((self.buffer_size, max_length, self.num_agents)).to(
            self.device
        )
        reward = torch.zeros((self.buffer_size, max_length, self.num_agents)).to(self.device)
        states = torch.zeros((self.buffer_size, max_length, self.state_space)).to(
            self.device
        )
        done = torch.zeros((self.buffer_size, max_length)).to(self.device)
        mask = torch.zeros(self.buffer_size, max_length, dtype=torch.bool).to(
            self.device
        )
        for i in range(self.buffer_size):
            length = lengths[i]
            obs[i, :length] = self.episodes[i]["obs"]
            avail_actions[i, :length] = self.episodes[i]["avail_actions"]
            actions[i, :length] = self.episodes[i]["actions"]
            log_probs[i, :length] = self.episodes[i]["log_prob"]
            reward[i, :length] = self.episodes[i]["reward"]
            states[i, :length] = self.episodes[i]["states"]
            done[i, :length] = self.episodes[i]["done"]
            mask[i, :length] = 1
        if self.normalize_reward:
            mu = torch.mean(reward[mask])
            std = torch.std(reward[mask])
            reward[mask.bool()] = (reward[mask] - mu) / (std + 1e-6)
        self.episodes = [None] * self.buffer_size
        return (
            obs.float(),
            actions.long(),
            log_probs.float(),
            reward.float(),
            states.float(),
            avail_actions.bool(),
            done.float(),
            mask,
        )


class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, output_dim) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        for i in range(num_layer):
            self.layers.append(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            )
        self.layers.append(nn.Sequential(nn.Linear(hidden_dim, output_dim)))

    def act(self, x, avail_action=None):
        logits = self.logits(x, avail_action)
        distribution = Categorical(logits=logits)
        action = distribution.sample()
        return action, distribution.log_prob(action)

    def logits(self, x, avail_action=None):
        for layer in self.layers:
            x = layer(x)
        if avail_action is not None:
            x = x.masked_fill(~avail_action, -1e9)
        return x


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, output_dim=1):
        super().__init__()
        self.output_dim = output_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        for i in range(num_layer):
            self.layers.append(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            )
        self.layers.append(nn.Sequential(nn.Linear(hidden_dim, output_dim)))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        if self.output_dim == 1:
            return x.squeeze(-1)
        return x


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def environment(env_type, env_name, env_family, agent_ids, kwargs):
    if env_type == "pz":
        from env.pettingzoo_wrapper import PettingZooWrapper

        env = PettingZooWrapper(
            family=env_family, env_name=env_name, agent_ids=agent_ids, **kwargs
        )
    elif env_type == "smaclite":
        from env.smaclite_wrapper import SMACliteWrapper

        env = SMACliteWrapper(map_name=env_name, agent_ids=agent_ids, **kwargs)
    elif env_type == "lbf":
        from env.lbf import LBFWrapper

        env = LBFWrapper(map_name=env_name, agent_ids=agent_ids, **kwargs)
    else:
        raise ValueError(f"Unsupported env_type: {env_type}")

    return env

def norm_d(grads, d):
    norms = [torch.linalg.vector_norm(g.detach(), d) for g in grads]
    total_norm_d = torch.linalg.vector_norm(torch.tensor(norms), d)
    return total_norm_d


def compute_approx_nashconv(args, kwargs, actors, device, n_agents):
    base_eval_env = environment(
        env_type=args.env_type,
        env_name=args.env_name,
        env_family=args.env_family,
        agent_ids=args.agent_ids,
        kwargs=kwargs,
    )
    try:
        base_utilities = evaluate_policy_utilities(
            base_eval_env,
            actors,
            device,
            args.approx_nashconv_eval_episodes,
            deterministic=args.approx_nashconv_deterministic_eval,
        )
    finally:
        base_eval_env.close()

    gaps = []
    for agent_idx in range(n_agents):
        br_actor = train_best_response_actor(args, kwargs, actors, agent_idx, device)
        br_eval_env = environment(
            env_type=args.env_type,
            env_name=args.env_name,
            env_family=args.env_family,
            agent_ids=args.agent_ids,
            kwargs=kwargs,
        )
        try:
            br_utilities = evaluate_policy_utilities(
                br_eval_env,
                actors,
                device,
                args.approx_nashconv_eval_episodes,
                deterministic=args.approx_nashconv_deterministic_eval,
                override_actor=br_actor,
                override_agent_idx=agent_idx,
            )
        finally:
            br_eval_env.close()
        gaps.append(max(0.0, float(br_utilities[agent_idx] - base_utilities[agent_idx])))

    return float(np.sum(gaps)), gaps, base_utilities



def build_actor(env, args, device):
    return Actor(
        input_dim=env.get_obs_size(),
        hidden_dim=args.actor_hidden_dim,
        num_layer=args.actor_num_layers,
        output_dim=env.get_action_size(),
    ).to(device)


def build_actors(env, args, device):
    return [build_actor(env, args, device) for _ in range(env.n_agents)]


def build_critic(env, args, device):
    return Critic(
        input_dim=env.get_state_size(),
        hidden_dim=args.critic_hidden_dim,
        num_layer=args.critic_num_layers,
        output_dim=env.n_agents,
    ).to(device)


def get_step_utilities(env, reward):
    if hasattr(env, "get_last_reward_vector"):
        reward_vector = np.asarray(env.get_last_reward_vector(), dtype=np.float32)
        if reward_vector.shape[0] == env.n_agents:
            return reward_vector
    reward_array = np.asarray(reward, dtype=np.float32)
    if reward_array.ndim == 0:
        return np.full(env.n_agents, float(reward_array), dtype=np.float32)
    if reward_array.shape[0] == env.n_agents:
        return reward_array
    return np.full(env.n_agents, float(reward_array.mean()), dtype=np.float32)


def get_team_reward(env, reward):
    return float(get_step_utilities(env, reward).sum())


def hybrid_policy_action(
    actors,
    obs,
    avail_actions,
    device,
    deterministic=False,
    override_actor=None,
    override_agent_idx=None,
):
    obs_tensor = torch.from_numpy(obs).float().to(device)
    avail_tensor = torch.from_numpy(avail_actions).bool().to(device)
    actions = []
    log_probs = []
    entropies = []
    for agent_idx, actor in enumerate(actors):
        current_actor = actor
        if override_actor is not None and override_agent_idx == agent_idx:
            current_actor = override_actor
        logits = current_actor.logits(
            obs_tensor[agent_idx].unsqueeze(0),
            avail_tensor[agent_idx].unsqueeze(0),
        ).squeeze(0)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        actions.append(action)
        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())
    return torch.stack(actions), torch.stack(log_probs), torch.stack(entropies)


def evaluate_policy_utilities(
    eval_env,
    actors,
    device,
    num_episodes,
    deterministic=False,
    override_actor=None,
    override_agent_idx=None,
):
    episodic_utilities = []
    for _ in range(num_episodes):
        obs, _ = eval_env.reset()
        done = False
        truncated = False
        ep_utility = np.zeros(eval_env.n_agents, dtype=np.float64)
        while not done and not truncated:
            avail_action = eval_env.get_avail_actions()
            with torch.no_grad():
                actions, _, _ = hybrid_policy_action(
                    actors,
                    obs,
                    avail_action,
                    device,
                    deterministic=deterministic,
                    override_actor=override_actor,
                    override_agent_idx=override_agent_idx,
                )
            obs, reward, done, truncated, _ = eval_env.step(actions.cpu().numpy())
            ep_utility += get_step_utilities(eval_env, reward)
        episodic_utilities.append(ep_utility)
    return np.mean(episodic_utilities, axis=0)


def train_best_response_actor(args, kwargs, actors, agent_idx, device):
    br_env = environment(
        env_type=args.env_type,
        env_name=args.env_name,
        env_family=args.env_family,
        agent_ids=args.agent_ids,
        kwargs=kwargs,
    )
    br_actor = build_actor(br_env, args, device)
    br_actor.load_state_dict(actors[agent_idx].state_dict())
    optimizer_cls = getattr(optim, args.optimizer)
    br_optimizer = optimizer_cls(br_actor.parameters(), lr=args.approx_nashconv_learning_rate)
    try:
        for _ in range(args.approx_nashconv_br_updates):
            batch_log_probs, batch_returns, batch_entropies = [], [], []
            for _ in range(args.approx_nashconv_br_episodes):
                obs, _ = br_env.reset()
                done = False
                truncated = False
                ep_rewards, ep_log_probs, ep_entropies = [], [], []
                while not done and not truncated:
                    actions, log_probs, entropies = hybrid_policy_action(
                        actors,
                        obs,
                        br_env.get_avail_actions(),
                        device,
                        deterministic=False,
                        override_actor=br_actor,
                        override_agent_idx=agent_idx,
                    )
                    obs, reward, done, truncated, _ = br_env.step(actions.cpu().numpy())
                    ep_rewards.append(float(get_step_utilities(br_env, reward)[agent_idx]))
                    ep_log_probs.append(log_probs[agent_idx])
                    ep_entropies.append(entropies[agent_idx])
                last_return = 0.0
                ep_returns = []
                for reward_t in reversed(ep_rewards):
                    last_return = reward_t + args.gamma * last_return
                    ep_returns.append(last_return)
                batch_returns.extend(reversed(ep_returns))
                batch_log_probs.extend(ep_log_probs)
                batch_entropies.extend(ep_entropies)
            if not batch_log_probs:
                continue
            returns = torch.tensor(batch_returns, dtype=torch.float32, device=device)
            if returns.numel() > 1:
                returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-6)
            loss = -(torch.stack(batch_log_probs) * returns).mean()
            loss -= args.entropy_coef * torch.stack(batch_entropies).mean()
            br_optimizer.zero_grad()
            loss.backward()
            if args.clip_gradients > 0:
                torch.nn.utils.clip_grad_norm_(br_actor.parameters(), max_norm=args.clip_gradients)
            br_optimizer.step()
    finally:
        br_env.close()
    return br_actor


if __name__ == "__main__":
    args = tyro.cli(Args)
    # Set random seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    ## import the environment
    kwargs = {}  # {"render_mode":'human',"shared_reward":False}
    env = environment(
        env_type=args.env_type,
        env_name=args.env_name,
        env_family=args.env_family,
        agent_ids=args.agent_ids,
        kwargs=kwargs,
    )
    eval_env = environment(
        env_type=args.env_type,
        env_name=args.env_name,
        env_family=args.env_family,
        agent_ids=args.agent_ids,
        kwargs=kwargs,
    )

    ## Initialize the actor, critic and target-critic networks
    actors = build_actors(env, args, device)
    critic = build_critic(env, args, device)
    Optimizer = getattr(optim, args.optimizer)
    actor_optimizers = [
        Optimizer(actor.parameters(), lr=args.learning_rate_actor) for actor in actors
    ]
    critic_optimizer = Optimizer(critic.parameters(), lr=args.learning_rate_critic)

    time_token = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.env_type}__{args.env_name}__{time_token}"
    if args.use_wnb:
        import wandb

        wandb.init(
            project=args.wnb_project,
            entity=args.wnb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=f"IPPO-{run_name}",
        )
    writer = SummaryWriter(f"runs/IPPO-{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    rb = RolloutBuffer(
        buffer_size=args.batch_size,
        obs_space=env.get_obs_size(),
        state_space=env.get_state_size(),
        action_space=env.get_action_size(),
        num_agents=env.n_agents,
        normalize_reward=args.normalize_reward,
        device=device,
    )
    ep_rewards = []
    ep_lengths = []
    ep_stats = []
    training_step = 0
    num_episodes = 0
    step = 0
    while step < args.total_timesteps:
        num_episode = 0
        while num_episode < args.batch_size:
            episode = {
                "obs": [],
                "actions": [],
                "log_prob": [],
                "reward": [],
                "states": [],
                "done": [],
                "avail_actions": [],
            }
            obs, _ = env.reset()
            ep_reward, ep_length = 0, 0
            done, truncated = False, False
            while not done and not truncated:
                avail_action = env.get_avail_actions()
                state = env.get_state()
                with torch.no_grad():
                    actions, log_probs, _ = hybrid_policy_action(
                        actors,
                        obs,
                        avail_action,
                        device,
                    )
                next_obs, reward, done, truncated, infos = env.step(
                    actions.cpu().numpy()
                )
                step_utilities = get_step_utilities(env, reward)
                ep_reward += float(step_utilities.sum())
                ep_length += 1
                step += 1

                episode["obs"].append(obs)
                episode["actions"].append(actions.cpu())
                episode["log_prob"].append(log_probs.cpu())
                episode["reward"].append(step_utilities)
                episode["done"].append(done)
                episode["avail_actions"].append(avail_action)
                episode["states"].append(state)

                obs = next_obs

            rb.add(episode)
            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_length)
            if args.env_type in ["smaclite", "lbf"]:
                ep_stats.append(infos)
            num_episode += 1

        num_episodes += args.batch_size
        ## logging
        if len(ep_rewards) > args.log_every:
            writer.add_scalar("rollout/ep_reward", np.mean(ep_rewards), step)
            writer.add_scalar("rollout/ep_length", np.mean(ep_lengths), step)
            writer.add_scalar("rollout/num_episodes", num_episodes, step)
            if args.env_type == "smaclite":
                writer.add_scalar(
                    "rollout/battle_won",
                    np.mean([info["battle_won"] for info in ep_stats]),
                    step,
                )
            elif args.env_type == "lbf":
                writer.add_scalar(
                    "rollout/food_collected_ratio",
                    np.mean([info.get("food_collected_ratio", 0.0) for info in ep_stats]),
                    step,
                )
            ep_rewards = []
            ep_lengths = []
            ep_stats = []
        ## Collate episodes in buffer into single batch
        (
            b_obs,
            b_actions,
            b_log_probs,
            b_reward,
            b_states,
            b_avail_actions,
            b_done,
            b_mask,
        ) = rb.get_batch()

        # Compute the advantage
        #####  Compute TD(λ) using "Reconciling λ-Returns with Experience Replay"(https://arxiv.org/pdf/1810.09967 Equation 3)
        #####  Compute the advantage using A(s,a) = λ-Returns -V(s), see page 47 in David Silver's lecture n 4 (https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/lecture-4-model-free-prediction-.pdf)
        return_lambda = torch.zeros_like(b_actions).float().to(device)
        advantages = torch.zeros_like(b_actions).float().to(device)
        # a Batched version
        # with torch.no_grad():
        #     for ep_idx in range(return_lambda.size(0)):
        #         next_value = critic(x=b_obs[ep_idx])
        #         next_value[~b_mask[ep_idx]] = 0
        #         ep_len = b_mask[ep_idx].sum()
        #         next_value = torch.cat((next_value, torch.zeros((1, env.n_agents))))
        #         last_return_lambda = 0
        #         for t in reversed(range(ep_len)):
        #             return_lambda[ep_idx, t] = last_return_lambda = b_reward[
        #                 ep_idx, t
        #             ] + args.gamma * (
        #                 args.td_lambda * last_return_lambda
        #                 + (1 - args.td_lambda) * next_value[t + 1]
        #             )
        #             advantages[ep_idx, t] = return_lambda[ep_idx, t] - next_value[t]
        with torch.no_grad():
            for ep_idx in range(return_lambda.size(0)):
                ep_len = b_mask[ep_idx].sum()
                last_return_lambda = 0
                for t in reversed(range(ep_len)):
                    if t == (ep_len - 1):
                        next_value = 0
                    else:
                        next_value = critic(x=b_states[ep_idx, t + 1])
                    return_lambda[ep_idx, t] = last_return_lambda = b_reward[
                        ep_idx, t
                    ] + args.gamma * (
                        args.td_lambda * last_return_lambda
                        + (1 - args.td_lambda) * next_value
                    )
                    advantages[ep_idx, t] = return_lambda[ep_idx, t] - critic(
                        x=b_states[ep_idx, t]
                    )

        # training loop
        if args.normalize_advantage:
            adv_mu = advantages.mean(dim=-1)[b_mask].mean()
            adv_std = advantages.mean(dim=-1)[b_mask].std()
            advantages = (advantages - adv_mu) / adv_std
        if args.normalize_return:
            ret_mu = return_lambda.mean(dim=-1)[b_mask].mean()
            ret_std = return_lambda.mean(dim=-1)[b_mask].std()
            return_lambda = (return_lambda - ret_mu) / ret_std
        actor_losses = []
        critic_losses = []
        entropies_bonuses = []
        kl_divergences = []
        actor_gradients = []
        critic_gradients = []
        clipped_ratios = []
        for _ in range(args.epochs):
            actor_loss = 0.0
            critic_loss = 0
            entropies = 0.0
            kl_divergence = 0.0
            clipped_ratio = 0.0
            current_actor_losses = []
            current_actor_gradients = []
            for agent_idx, actor in enumerate(actors):
                agent_actor_loss = 0
                agent_entropy = 0
                agent_kl_divergence = 0
                agent_clipped_ratio = 0
                for t in range(b_obs.size(1)):
                    current_logits = actor.logits(
                        x=b_obs[:, t, agent_idx],
                        avail_action=b_avail_actions[:, t, agent_idx],
                    )
                    current_dist = Categorical(logits=current_logits)
                    current_logprob = current_dist.log_prob(b_actions[:, t, agent_idx])
                    log_ratio = current_logprob - b_log_probs[:, t, agent_idx]
                    ratio = torch.exp(log_ratio)
                    pg_loss1 = advantages[:, t, agent_idx] * ratio
                    pg_loss2 = advantages[:, t, agent_idx] * torch.clamp(
                        ratio, 1 - args.ppo_clip, 1 + args.ppo_clip
                    )
                    if b_mask[:, t].any().item():
                        pg_loss = torch.min(
                            pg_loss1[b_mask[:, t]], pg_loss2[b_mask[:, t]]
                        ).mean()
                        entropy_loss = current_dist.entropy()[b_mask[:, t]].mean()
                        b_kl_divergence = ((ratio - 1) - log_ratio)[b_mask[:, t]].mean()
                        b_clipped_ratio = (
                            ((ratio - 1.0).abs() > args.ppo_clip)[b_mask[:, t]]
                            .float()
                            .mean()
                        )
                    else:
                        pg_loss = torch.tensor(0.0, device=device)
                        entropy_loss = torch.tensor(0.0, device=device)
                        b_kl_divergence = torch.tensor(0.0, device=device)
                        b_clipped_ratio = torch.tensor(0.0, device=device)
                    agent_entropy += entropy_loss
                    agent_actor_loss += -pg_loss - args.entropy_coef * entropy_loss
                    agent_kl_divergence += b_kl_divergence
                    agent_clipped_ratio += b_clipped_ratio
                agent_actor_loss /= b_mask.sum()
                agent_entropy /= b_mask.sum()
                agent_kl_divergence /= b_mask.sum()
                agent_clipped_ratio /= b_mask.sum()
                actor_optimizers[agent_idx].zero_grad()
                agent_actor_loss.backward()
                actor_gradient = norm_d([p.grad for p in actor.parameters()], 2)
                if args.clip_gradients > 0:
                    torch.nn.utils.clip_grad_norm_(
                        actor.parameters(), max_norm=args.clip_gradients
                    )
                actor_optimizers[agent_idx].step()
                current_actor_losses.append(agent_actor_loss.item())
                current_actor_gradients.append(actor_gradient.item())
                actor_loss += agent_actor_loss.item()
                entropies += agent_entropy.item()
                kl_divergence += agent_kl_divergence.item()
                clipped_ratio += agent_clipped_ratio.item()

            for t in range(b_states.size(1)):
                current_values = critic(x=b_states[:, t])
                value_loss = F.mse_loss(
                    current_values[b_mask[:, t]], return_lambda[:, t][b_mask[:, t]]
                ) * (b_mask[:, t].sum())
                critic_loss += value_loss
            critic_loss /= b_mask.sum()
            critic_optimizer.zero_grad()
            critic_loss.backward()
            training_step += 1
            critic_gradient = norm_d([p.grad for p in critic.parameters()], 2)
            if args.clip_gradients > 0:
                torch.nn.utils.clip_grad_norm_(
                    critic.parameters(), max_norm=args.clip_gradients
                )
            critic_optimizer.step()

            actor_losses.append(np.mean(current_actor_losses))
            critic_losses.append(critic_loss.item())
            entropies_bonuses.append(entropies / env.n_agents)
            kl_divergences.append(kl_divergence / env.n_agents)
            actor_gradients.append(np.mean(current_actor_gradients))
            critic_gradients.append(critic_gradient.item())
            clipped_ratios.append(clipped_ratio / env.n_agents)

        writer.add_scalar("train/critic_loss", np.mean(critic_losses), step)
        writer.add_scalar("train/actor_loss", np.mean(actor_losses), step)
        writer.add_scalar("train/entropy", np.mean(entropies_bonuses), step)
        writer.add_scalar("train/kl_divergence", np.mean(kl_divergences), step)
        writer.add_scalar("train/clipped_ratios", np.mean(clipped_ratios), step)
        writer.add_scalar("train/actor_gradients", np.mean(actor_gradients), step)
        writer.add_scalar("train/critic_gradients", np.mean(critic_gradients), step)
        writer.add_scalar("train/num_updates", training_step, step)

        # Explicitly release large training tensors so Python's GC can reclaim
        # memory before the next rollout collection.
        del b_obs, b_actions, b_log_probs, b_reward, b_states
        del b_avail_actions, b_done, b_mask, return_lambda, advantages
        gc.collect()

        if (training_step / args.epochs) % args.eval_steps == 0:
            eval_obs, _ = eval_env.reset()
            eval_ep = 0
            eval_ep_reward = []
            eval_ep_length = []
            eval_ep_stats = []
            current_reward = 0
            current_ep_length = 0
            while eval_ep < args.num_eval_ep:
                with torch.no_grad():
                    actions, _, _ = hybrid_policy_action(
                        actors,
                        eval_obs,
                        eval_env.get_avail_actions(),
                        device,
                    )
                next_obs_, reward, done, truncated, infos = eval_env.step(
                    actions.cpu().numpy()
                )
                current_reward += get_team_reward(eval_env, reward)
                current_ep_length += 1
                eval_obs = next_obs_
                if done or truncated:
                    eval_obs, _ = eval_env.reset()
                    eval_ep_reward.append(current_reward)
                    eval_ep_length.append(current_ep_length)
                    eval_ep_stats.append(infos)
                    current_reward = 0
                    current_ep_length = 0
                    eval_ep += 1
            writer.add_scalar("eval/ep_reward", np.mean(eval_ep_reward), step)
            writer.add_scalar("eval/std_ep_reward", np.std(eval_ep_reward), step)
            writer.add_scalar("eval/ep_length", np.mean(eval_ep_length), step)
            if args.env_type == "smaclite":
                writer.add_scalar(
                    "eval/battle_won",
                    np.mean(np.mean([info["battle_won"] for info in eval_ep_stats])),
                    step,
                )
            elif args.env_type == "lbf":
                writer.add_scalar(
                    "eval/food_collected_ratio",
                    np.mean([info.get("food_collected_ratio", 0.0) for info in eval_ep_stats]),
                    step,
                )
            if args.approx_nashconv:
                approx_nashconv, approx_nash_gaps, base_utilities = compute_approx_nashconv(
                    args, kwargs, actors, device, env.n_agents
                )
                writer.add_scalar("eval/approx_nashconv", approx_nashconv, step)
                writer.add_scalar(
                    "eval/approx_nashconv_max_gap", np.max(approx_nash_gaps), step
                )
                for agent_idx, gap in enumerate(approx_nash_gaps):
                    writer.add_scalar(
                        f"eval/approx_nash_gap_agent_{agent_idx}", gap, step
                    )
                    writer.add_scalar(
                        f"eval/base_utility_agent_{agent_idx}",
                        base_utilities[agent_idx],
                        step,
                    )

    writer.close()
    if args.use_wnb:
        wandb.finish()
    env.close()
    eval_env.close()
