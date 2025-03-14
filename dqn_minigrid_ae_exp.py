# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import minigrid
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import DictReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from utils import TransposeImageWrapper, StateRecordingWrapper, SafetyAwareObservationWrapper


@dataclass
class Args:
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
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""
    reseed: bool = False
    """whether to fix seed on environment reset"""
    plot_state_heatmap: bool = True
    """whether to plot state heatmap"""

    # VAE arguments
    ae_dim: int = 50
    """the dimensionality of the latent space"""
    ae_batch_size: int = 32
    """the batch size to use for VAE training"""
    vae_beta: float = 0.0001
    """KL coefficient in VAE"""
    reconstruct_loss_coef: float = 100
    """reconstruction loss coefficient in VAE objective"""
    latent_dist_coef: float = 0.01
    """latent distance loss coefficient in AE objective"""
    max_grad_norm: float = 0.5
    """the maximum norm for gradient clipping"""
    ae_buffer_size: int = 50_000
    """buffer size for training VAE"""
    save_ae_training_data_freq: int = -1
    """save training AE data buffer every n environment steps."""
    save_sample_ae_reconstruction_every: int = 50_000
    """save sample reconstruction from AE every n environment steps."""
    deterministic_latent: bool = False
    """deterministically sample from VAE when inference."""
    vae_training_frequency: int = 5
    """the frequency of VAE training"""
    save_final_buffer: bool = False
    """save the buffer at the end of training."""
    ae_warmup_steps: int = 1000
    """warmup phase for VAE; intrinsic rewards are not considered in this period."""
    min_latent_distance: float = 25.0
    """min latent distance used in AE training objective"""


def make_env(env_id, seed, idx, capture_video, run_name):
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
            if args.reseed:
                env = minigrid.wrappers.ReseedWrapper(env, seeds=(args.seed,))
            if args.plot_state_heatmap:
                env = StateRecordingWrapper(env)
            env = SafetyAwareObservationWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk


# Modified from SAC AE
# https://github.com/denisyarats/pytorch_sac_ae/blob/master/encoder.py#L11
# ===================================

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

        self.outputs = dict()

    def forward_conv(self, obs):
        obs = self.resize(obs)
        obs = obs.float()
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
        self.outputs['fc'] = h_fc

        # h_norm = self.ln(h_fc)
        # self.outputs['ln'] = h_norm

        self.outputs['latent'] = h_fc

        return h_fc


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


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
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
            tags=['dqn', 'vae', 'exp']
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
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs, input_dim=args.ae_dim).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs, input_dim=args.ae_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())

    # VAE
    encoder = PixelEncoder(envs.single_observation_space['image'].shape, args.ae_dim).to(device)
    decoder = PixelDecoder(envs.single_observation_space['image'].shape, args.ae_dim).to(device)
    encoder_optim = optim.Adam(encoder.parameters(), lr=args.learning_rate, eps=1e-5)
    decoder_optim = optim.Adam(decoder.parameters(), lr=args.learning_rate, eps=1e-5)

    rb = DictReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    unsafe_rb = DictReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    start_time = time.time()
    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    if args.plot_state_heatmap:
        state_cnt_recorder = envs.get_attr('state_cnt_recorder')[0]

    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps,
                                  global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            obs_embedding = encoder(torch.Tensor(obs['image']).to(device))
            q_values = q_network(obs_embedding)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for idx, info in enumerate(infos["final_info"]):
                if info and 'unsafe' in info:
                    obs['unsafe'][idx] = info['unsafe']
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        if obs['unsafe'][0] == 1:
            unsafe_rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        else:
            rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    next_obs_embeddings = encoder(data.next_observations['image'])
                    target_actions = q_network(next_obs_embeddings).argmax(dim=1, keepdim=True)
                    target_max = target_network(next_obs_embeddings).gather(1, target_actions).flatten()
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                obs_embeddings = encoder(data.observations['image'])
                old_val = q_network(obs_embeddings).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                encoder_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
                optimizer.step()
                encoder_optim.step()

            if global_step % args.vae_training_frequency == 0:
                unsafe_batch_size = min(args.ae_batch_size // 2, unsafe_rb.size())
                data = rb.sample(args.ae_batch_size - unsafe_batch_size)
                data_unsafe = unsafe_rb.sample(unsafe_batch_size)

                data_observations = torch.cat([data.observations['image'], data_unsafe.observations['image']], dim=0)
                data_safety = torch.cat([data.observations['unsafe'], data_unsafe.observations['unsafe']], dim=0)
                unsafe = 2 * data_safety.float() - 1
                latent = encoder(data_observations)
                reconstruction = decoder(latent)

                assert encoder.outputs['obs'].shape == reconstruction.shape
                reconstruct_loss = F.mse_loss(reconstruction, encoder.outputs['obs'])
                latent_norm = (latent ** 2).sum(dim=-1).mean()

                latent_dist = torch.cdist(latent, latent)
                dist_weights = unsafe @ unsafe.T
                assert dist_weights.shape == latent_dist.shape
                n_positive = torch.sum(dist_weights > 0)
                n_negative = torch.sum(dist_weights < 0)
                weights = torch.where(dist_weights > 0, 1 / n_positive, 1 / n_negative)
                min_dist = torch.where(dist_weights >= 0, 0.0, args.min_latent_distance)
                latent_dist_squared = torch.where(dist_weights >= 0, latent_dist ** 2, torch.clamp(args.min_latent_distance - latent_dist ** 2, min=0.0))
                latent_dist_loss = 0.5 * torch.sum(latent_dist_squared * weights)
                # latent_dist_loss = 0.5 * torch.sum(((latent_dist - min_dist) ** 2) * weights)
                loss = args.reconstruct_loss_coef * reconstruct_loss + args.latent_dist_coef * latent_dist_loss

                if global_step % 100 == 0:
                    writer.add_scalar("losses/ae_loss", loss.item(), global_step)
                    writer.add_scalar("losses/ae_reconstruction_loss", reconstruct_loss.item(), global_step)
                    writer.add_scalar("losses/ae_latent_norm", latent_norm.item(), global_step)
                    writer.add_scalar("losses/latent_dist_loss", latent_dist_loss.item(), global_step)
                    writer.add_scalar("losses/positive_dist_mean", latent_dist[dist_weights > 0].mean().item(),
                                      global_step)
                    writer.add_scalar("losses/negative_dist_mean", latent_dist[dist_weights < 0].mean().item(),
                                      global_step)

                if global_step % args.save_sample_ae_reconstruction_every == 0:
                    # AE reconstruction
                    save_reconstruction = reconstruction[0].detach()
                    save_reconstruction = (save_reconstruction * 128 + 128).clip(0, 255).cpu()

                    # AE target
                    ae_target = encoder.outputs['obs'][0].detach()
                    ae_target = (ae_target * 128 + 128).clip(0, 255).cpu()

                    # log
                    writer.add_image('image/AE reconstruction', save_reconstruction.type(torch.uint8), global_step)
                    writer.add_image('image/original', data.observations['image'][0].cpu().type(torch.uint8),
                                     global_step)
                    writer.add_image('image/AE target', ae_target.type(torch.uint8), global_step)

                encoder_optim.zero_grad()
                decoder_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
                nn.utils.clip_grad_norm_(decoder.parameters(), args.max_grad_norm)
                encoder_optim.step()
                decoder_optim.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

            if global_step % (args.total_timesteps // 10) == 0:
                if args.plot_state_heatmap:
                    writer.add_figure("figures/state_heatmap",
                                      state_cnt_recorder.get_figure_log_scale(), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")

        ae_model_path = f"runs/{run_name}/{args.exp_name}_ae.pt"
        torch.save({
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict()
        }, ae_model_path)
        print(f"auto-encoder saved to {ae_model_path}")

        # from cleanrl_utils.evals.dqn_eval import evaluate
        #
        # episodic_returns = evaluate(
        #     model_path,
        #     make_env,
        #     args.env_id,
        #     eval_episodes=10,
        #     run_name=f"{run_name}-eval",
        #     Model=QNetwork,
        #     device=device,
        #     epsilon=0.05,
        # )
        # for idx, episodic_return in enumerate(episodic_returns):
        #     writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    if args.plot_state_heatmap:
        run_path = f"runs/{run_name}"
        state_cnt_recorder.save_to(f"{run_path}/state_heatmap.npz")
        state_cnt_recorder.get_figure_log_scale()
        import matplotlib.pyplot as plt

        plt.savefig(f"{run_path}/state_heatmap.svg", format="svg",
                    transparent=True)

    envs.close()
    writer.close()
