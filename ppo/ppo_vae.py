# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import minigrid.wrappers
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from lnt.lnt import SafetyWrapper
from QLearningAgent import QLearningAgent
from utils import TransposeImageWrapper
from QLearningAgent import Args as QLearningArgs

@dataclass
class Args:
    qlearning: QLearningArgs
    """Q-learning configuration"""
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    min_q: float = -0.2
    """the min q value used in LeaveNoTrace"""
    reseed: bool = False
    """whether to regenerate the environment with the same seed"""
    ae_dim: int = 50
    """ae_dim"""
    ae_batch_size: int = 32
    """ae_batch_size"""
    beta: float = 0.0001
    """L2 norm of the latent vectors"""
    ae_buffer_size: int = 100_000
    """buffer size for training ae, recommend less than 200k"""
    save_ae_training_data_freq: int = -1
    """Save training AE data buffer every env steps"""
    save_sample_AE_reconstruction_every: int = 200_000
    """Save sample reconstruction from AE every env steps"""
    weight_decay: float = 0.
    """L2 norm of the weight vectors of decoder"""

def make_env(env_id, idx, capture_video, run_name, args=None, writer=None, device=None):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)

        if "MiniGrid" in env_id:
            env = minigrid.wrappers.RGBImgObsWrapper(env)
            env = minigrid.wrappers.ImgObsWrapper(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = TransposeImageWrapper(env)
            # env = DeathLogWrapper(env)
            if args.reseed:
                env = minigrid.wrappers.ReseedWrapper(env, seeds=(args.seed,))
            # Safety Wrapper
            q_learning_agent = QLearningAgent(env, args.qlearning, writer, device)
            def reset_reward_fn(env, obs, action):
                return float(reset_done_fn(env, obs))
            def reset_done_fn(env, obs):
                return np.array_equal(env.unwrapped.agent_pos, np.array([1, 1]))

            env = SafetyWrapper(env, q_learning_agent, reset_reward_fn, reset_done_fn, args.min_q)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


OUT_DIM = {2: 39, 4: 35, 6: 31}


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""

    def __init__(self, obs_shape, feature_dim, num_layers=4, num_filters=32):
        super().__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        from torchvision.transforms import Resize
        self.resize = Resize((84, 84))  # Input image is resized to []

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = OUT_DIM[num_layers]
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        # self.ln = nn.LayerNorm(self.feature_dim)
        self.fc_mu = nn.Linear(self.feature_dim, self.feature_dim)
        self.fc_var = nn.Linear(self.feature_dim, self.feature_dim)

        self.outputs = dict()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        obs = self.resize(obs)
        obs = (obs - 128) / 128.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        # self.outputs['fc'] = h_fc
        h_fc = torch.relu(h_fc)

        # h_norm = self.ln(h_fc)
        # self.outputs['ln'] = h_norm

        self.outputs['latent'] = h_fc

        return self.fc_mu(h_fc), self.fc_var(h_fc)

    def sample(self, obs, deterministic=False):
        mu, logvar = self(obs)
        if deterministic:
            return mu, mu, logvar
        return self.reparameterize(mu, logvar), mu, logvar


class PixelDecoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers=4, num_filters=32):
        super().__init__()

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.out_dim = OUT_DIM[num_layers]

        self.fc = nn.Linear(
            feature_dim, num_filters * self.out_dim * self.out_dim
        )

        self.deconvs = nn.ModuleList()

        for i in range(self.num_layers - 1):
            self.deconvs.append(
                nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1)
            )
        self.deconvs.append(
            nn.ConvTranspose2d(
                num_filters, obs_shape[0], 3, stride=2, output_padding=1
            )
        )

        self.outputs = dict()

    def forward(self, h):
        h = torch.relu(self.fc(h))
        self.outputs['fc'] = h

        deconv = h.view(-1, self.num_filters, self.out_dim, self.out_dim)
        self.outputs['deconv1'] = deconv

        for i in range(0, self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))
            self.outputs['deconv%s' % (i + 1)] = deconv

        obs = self.deconvs[-1](deconv)
        obs = torch.tanh(obs)
        self.outputs['obs'] = obs

        return obs


# ===================================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, obs_shape, hidden_dim=64):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, detach_value=False, detach_policy=True):
        if detach_policy:
            logits = self.actor(x.detach())
        else:
            logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        if detach_value: x = x.detach()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class Episode:
    def __init__(self, env, embedding_dim, max_len=500, device='cpu'):
        self.max_len = max_len
        self.obs = torch.zeros((self.max_len, env.num_envs, embedding_dim)).to(device)
        # Index for each environment
        self.indx = torch.zeros((env.num_envs,), dtype=torch.long)
        self.device = device
        self.full = torch.zeros_like(self.indx, dtype=torch.bool)

    def add(self, embedding):
        # embedding: (n_env, embedding_dim)
        for env_idx in range(self.obs.shape[2]):
            self.obs[self.indx[env_idx], env_idx] = embedding[env_idx]
        self.indx = (self.indx + 1) % self.max_len
        for i in range(len(self.indx)):
            if self.indx[i] == 0:
                self.full[i] = True

    def reset_at(self, env_idx):
        self.indx[env_idx] = 0
        self.full[env_idx] = False

    def get_state(self, i):
        if self.full[i]:
            return self.obs[:, i]
        return self.obs[:self.indx[i], i]

    def get_states(self):
        res = [self.get_state(i) for i in range(len(self.indx))]
        return res


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args, writer, device) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs, obs_shape=args.ae_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # Auto-encoder hyperparameters
    ae_dim = args.ae_dim
    beta = args.beta
    encoder, decoder = (
        PixelEncoder(envs.single_observation_space.shape, ae_dim).to(device),
        PixelDecoder(envs.single_observation_space.shape, ae_dim).to(device)
    )
    encoder_optim = optim.Adam(encoder.parameters(), lr=args.learning_rate, eps=1e-5)
    decoder_optim = optim.Adam(decoder.parameters(), lr=args.learning_rate,
                               eps=1e-5, weight_decay=args.weight_decay)

    args.ae_buffer_size = args.ae_buffer_size // args.num_envs

    buffer_ae = torch.zeros((args.ae_buffer_size, args.num_envs) + envs.single_observation_space.shape,
                            dtype=torch.uint8)
    next_obs_buffer_ae = torch.zeros((args.ae_buffer_size, args.num_envs) + envs.single_observation_space.shape,
                            dtype=torch.uint8)
    done_buffer = torch.zeros((args.ae_buffer_size, args.num_envs, 1), dtype=torch.bool)
    buffer_ae_indx = 0
    ae_buffer_is_full = False

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    prev_global_timestep = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    last_checkpoint = 0

    intrinsic_reward_measures = []
    latent_distance_measures = []

    """ record states in an episode for each parallel environment """
    episode_record = Episode(envs, embedding_dim=args.ae_dim,
                             max_len=args.window_size_episode, device=device)

    """ For visualization """
    if args.visualize_states:
        record_state = stateRecording(envs.envs[0])
        record_state.add_count_from_env(envs.envs[0])

    """ method for calculating Intrinsic reward """
    intrinsic_rw_fnc = reduce[args.reduce]

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            # Auto-encoder buffer
            buffer_ae[buffer_ae_indx] = next_obs.cpu()
            done_buffer[buffer_ae_indx] = next_done.cpu().reshape(done_buffer[buffer_ae_indx].shape)

            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                # Encode the observation
                next_obs_embedding = encoder.sample(next_obs, deterministic=args.deterministic_latent)[0]
                action, logprob, _, value = agent.get_action_and_value(next_obs_embedding)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            if args.visualize_states:
                record_state.add_count_from_env(envs.envs[0])
                if args.whiten_rewards:
                    """ hide reward from agents """
                    reward = np.zeros_like(reward)
            rewards[step] = torch.tensor(reward).to(device).view(-1) * args.reward_scale
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_obs_embedding = encoder.sample(next_obs, deterministic=args.deterministic_latent)[0]
            next_value = agent.get_value(next_obs_embedding).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    encoder(b_obs[mb_inds]), b_actions.long()[mb_inds],
                    detach_value=False, detach_policy=True
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                encoder_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
                optimizer.step()
                encoder_optim.step()

                # Train auto-encoder
                # Step 1: Sample from AE buffer
                current_ae_buffer_size = args.ae_buffer_size if ae_buffer_is_full else buffer_ae_indx
                ae_indx_batch = torch.randint(low=0, high=current_ae_buffer_size,
                                              size=(args.ae_batch_size,))
                ae_batch = buffer_ae[ae_indx_batch].float().to(device)
                # Flatten
                ae_batch = ae_batch.reshape((-1,) + envs.single_observation_space.shape)
                # Step 2: Forward pass
                latent = encoder(ae_batch)
                reconstruction = decoder(latent)
                assert encoder.outputs['obs'].shape == reconstruction.shape
                # Step 3: Calculate loss
                latent_norm = (latent ** 2).sum(dim=-1).mean()
                ae_loss = torch.nn.functional.mse_loss(reconstruction, encoder.outputs['obs']) + beta * latent_norm
                # Step 4: Backward
                encoder_optim.zero_grad()
                decoder_optim.zero_grad()
                ae_loss.backward()
                nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
                nn.utils.clip_grad_norm_(decoder.parameters(), args.max_grad_norm)
                encoder_optim.step()
                decoder_optim.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        # Auto-encoder logging
        if args.save_ae_training_data_freq > 0 and (global_step // args.num_envs) % (
                args.save_ae_training_data_freq // args.num_envs) == 0:
            os.makedirs("ae_data", exist_ok=True)
            file_path = os.path.join("ae_data", f"step_{global_step}.pt")
            torch.save(buffer_ae[:current_ae_buffer_size], file_path)

        # for some every step, save the image reconstructions of AE, for debugging purpose
        if (global_step - prev_global_timestep) >= args.save_sample_AE_reconstruction_every:
            # AE reconstruction
            save_reconstruction = reconstruction[0].detach()
            save_reconstruction = (save_reconstruction * 128 + 128).clip(0, 255).cpu()

            # AE target
            ae_target = encoder.outputs['obs'][0].detach()
            ae_target = (ae_target * 128 + 128).clip(0, 255).cpu()

            # log
            writer.add_image('image/AE reconstruction', save_reconstruction.type(torch.uint8), global_step)
            writer.add_image('image/original', ae_batch[0].cpu().type(torch.uint8), global_step)
            writer.add_image('image/AE target', ae_target.type(torch.uint8), global_step)
            prev_global_timestep = global_step
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()