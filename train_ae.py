import gymnasium as gym
import minigrid
from dataclasses import dataclass
import numpy as np
import random

import pandas as pd
import tyro
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os

import torch.optim as optim
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
import matplotlib.pyplot as plt

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""

    env_id: str = "MiniGrid-LavaCrossingS9N1-v0"
    """the id of the environment"""
    seed: int = 1
    """seed of the experiment"""
    ae_dim: int = 50
    """the dimensionality of the latent space"""
    batch_size: int = 16
    """the batch size used in training auto encoder"""
    epochs: int = 10
    """number of epochs to train auto encoder"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    reconstruct_loss_coef: float = 100
    """reconstruction loss coefficient in VAE objective"""
    max_grad_norm: float = 0.5
    """the maximum norm for gradient clipping"""
    latent_dist_coef: float = 0.01
    """latent distance loss coefficient in AE objective"""
    plotly: bool = False
    """whether to use interactive plotting with plotly"""
    save_model: bool = True
    """whether to save models"""

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
            [nn.Conv2d(obs_shape[2], num_filters, 3, stride=2)]
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
                num_filters, obs_shape[2], 3, stride=2, output_padding=1
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


if __name__ == '__main__':
    args = tyro.cli(Args)
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

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
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    env = gym.make(args.env_id, render_mode='rgb_array', max_steps=100)
    env = minigrid.wrappers.RGBImgObsWrapper(env)
    env = minigrid.wrappers.ImgObsWrapper(env)
    env = minigrid.wrappers.NoDeath(env, no_death_types=('lava',))
    env = minigrid.wrappers.ReseedWrapper(env, seeds=(args.seed,))
    print(env.observation_space.shape)

    obs, info = env.reset()
    grid = env.unwrapped.grid
    observations = np.zeros(((grid.width - 2) * (grid.height - 2) * 4,) + env.observation_space.shape, dtype=np.uint8)
    labels = np.zeros(((grid.width - 2) * (grid.height - 2) * 4,), dtype=np.int32)

    for i in range(grid.width):
        for j in range(grid.height):
            c = grid.get(i, j)
            if (c is not None and c.type != 'wall') or c is None:
                env.unwrapped.agent_pos = (i, j)
                for dir in range(4):
                    env.unwrapped.agent_dir = dir
                    obs = env.get_frame(highlight=env.unwrapped.highlight, tile_size=env.tile_size)
                    obs_idx = (i - 1) * (grid.height - 2) * 4 + (j - 1) * 4 + dir
                    observations[obs_idx] = obs

                    next_c = grid.get(*env.unwrapped.front_pos)
                    if c is not None and c.type == 'lava':
                        labels[obs_idx] = 1
                    elif next_c is not None and next_c.type == 'lava':
                        labels[obs_idx] = 1
                    else:
                        labels[obs_idx] = -1

                    assert obs_idx % 4 == dir
                    assert (obs_idx // 4) % (grid.height - 2) == j - 1
                    assert (obs_idx // (4 * (grid.height - 2))) == i - 1

    observations = observations.transpose(0, 3, 1, 2)
    all_observations = np.ascontiguousarray(observations)
    # Step 1: Convert NumPy arrays to PyTorch tensors
    data_tensor = torch.tensor(all_observations, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    # Step 2: Create a TensorDataset
    dataset = TensorDataset(data_tensor, labels_tensor)
    # Step 3: Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    encoder = PixelEncoder(env.observation_space.shape, args.ae_dim).to(device)
    decoder = PixelDecoder(env.observation_space.shape, args.ae_dim).to(device)
    encoder_optim = optim.Adam(encoder.parameters(), lr=args.learning_rate, eps=1e-5)
    decoder_optim = optim.Adam(decoder.parameters(), lr=args.learning_rate, eps=1e-5)

    # Example of iterating through the DataLoader
    global_step = 0
    for e in range(args.epochs):
        for observations, labels in dataloader:
            observations = observations.to(device)
            labels = labels.unsqueeze(1).float().to(device)

            latent = encoder(observations)
            reconstruction = decoder(latent)
            assert encoder.outputs['obs'].shape == reconstruction.shape
            reconstruct_loss = F.mse_loss(reconstruction, encoder.outputs['obs'])
            latent_norm = (latent ** 2).sum(dim=-1).mean()

            latent_dist = torch.cdist(latent, latent)
            dist_weights = labels @ labels.T
            assert dist_weights.shape == latent_dist.shape
            min_dist = torch.where(dist_weights > 0, 0.0, 10.0)
            latent_dist_loss = 0.5 * torch.mean((latent_dist - min_dist) ** 2)
            loss = args.reconstruct_loss_coef * reconstruct_loss + args.latent_dist_coef * latent_dist_loss

            writer.add_scalar("losses/ae_loss", loss.item(), global_step)
            writer.add_scalar("losses/ae_reconstruction_loss", reconstruct_loss.item(), global_step)
            writer.add_scalar("losses/ae_latent_norm", latent_norm.item(), global_step)
            writer.add_scalar("losses/latent_dist_loss", latent_dist_loss.item(), global_step)
            writer.add_scalar("losses/positive_dist_mean", latent_dist[dist_weights > 0].mean().item(),
                              global_step)
            writer.add_scalar("losses/negative_dist_mean", latent_dist[dist_weights < 0].mean().item(),
                              global_step)

            encoder_optim.zero_grad()
            decoder_optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
            nn.utils.clip_grad_norm_(decoder.parameters(), args.max_grad_norm)
            encoder_optim.step()
            decoder_optim.step()

            global_step += 1

        print(f"Epoch #{e+1}: {loss.item()}")
        if e % 50 == 0:
            # AE reconstruction
            save_reconstruction = reconstruction[0].detach()
            save_reconstruction = (save_reconstruction * 128 + 128).clip(0, 255).cpu()

            # AE target
            ae_target = encoder.outputs['obs'][0].detach()
            ae_target = (ae_target * 128 + 128).clip(0, 255).cpu()

            # log
            writer.add_image('image/AE reconstruction', save_reconstruction.type(torch.uint8), global_step)
            writer.add_image('image/original', observations[0].cpu().type(torch.uint8), global_step)
            writer.add_image('image/AE target', ae_target.type(torch.uint8), global_step)

    print("Training done! Visualizing embeddings...")
    embeddings = encoder(data_tensor.to(device))
    reconstruction = decoder(embeddings)
    assert encoder.outputs['obs'].shape == reconstruction.shape
    reconstruct_loss = torch.nn.functional.mse_loss(reconstruction, encoder.outputs['obs'])

    print(f'Reconstruction loss: {reconstruct_loss.item()}')
    cpu_embeddings = embeddings.detach().cpu().numpy()

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(cpu_embeddings)
    print(X_tsne.shape)
    print(tsne.kl_divergence_)
    df = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'])
    labels = ['safe' for _ in range(X_tsne.shape[0])]
    names = ['i, j, dir' for _ in range(X_tsne.shape[0])]

    for obs_idx in range(X_tsne.shape[0]):
        dir = obs_idx % 4
        j = (obs_idx // 4) % (grid.height - 2) + 1
        i = (obs_idx // (4 * (grid.height - 2))) + 1
        env.unwrapped.agent_pos = np.array((i, j))
        env.unwrapped.agent_dir = dir

        c = grid.get(i, j)
        next_c = grid.get(*env.unwrapped.front_pos)
        if c is not None and c.type == 'lava':
            labels[obs_idx] = 'death'

        if c is None and next_c is not None and next_c.type == 'lava':
            labels[obs_idx] = 'unsafe'

        names[obs_idx] = f'{i}, {j}, {dir}'

    df['label'] = labels
    df['name'] = names
    print(df['label'].value_counts())

    if args.plotly:
        import plotly.express as px

        fig = px.scatter(df, x='TSNE1', y='TSNE2', color='label', title='t-SNE visualization', hover_data=['name'])

        # Update layout to improve hover information
        fig.update_traces(marker=dict(size=10, opacity=0.8),
                          selector=dict(mode='markers+text'))

        fig.update_layout(
            hovermode='closest'
        )

        # Save plot to an HTML file
        fig.write_html(f'runs/{run_name}/tsne_visualization_with_index.html')

    else:
        ax = sns.scatterplot(data=df, x='TSNE1', y='TSNE2', hue='label')
        plt.title('t-SNE visualization')
        writer.add_figure("embedding_visualization",
                          plt.gcf(), global_step)
        plt.savefig(f'runs/{run_name}/t-SNE_{args.env_id}_{args.seed}.png')

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        encoder_model_path = f"runs/{run_name}/{args.exp_name}_encoder.pt"
        torch.save(encoder.state_dict(), encoder_model_path)
        print(f"encoder saved to {encoder_model_path}")

        decoder_model_path = f"runs/{run_name}/{args.exp_name}_decoder.pt"
        torch.save(decoder.state_dict(), decoder_model_path)
        print(f"decoder saved to {decoder_model_path}")