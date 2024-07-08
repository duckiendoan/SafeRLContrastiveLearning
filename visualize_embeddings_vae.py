import random
import gymnasium as gym
import minigrid
import tyro
from dataclasses import dataclass
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn


@dataclass
class Args:
    ae_path: str = './encoder.pth'
    """path to auto-encoder"""
    ae_dim: int = 50
    """the dimensionality of the latent space"""
    env_id: str = "MiniGrid-LavaCrossingS9N1-v0"
    """the id of the environment"""
    seed: int = 1
    """seed of the experiment"""
    plotly: bool = False
    """use plotly instead of seaborn"""


OUT_DIM = {2: 39, 4: 35, 6: 31}


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""

    def __init__(self, obs_shape, feature_dim, num_layers=4, num_filters=32):
        super().__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        from torchvision.transforms import Resize
        self.resize = Resize((84, 84), interpolation=0)  # Input image is resized to []

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[2], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = OUT_DIM[num_layers]
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        # self.ln = nn.LayerNorm(self.feature_dim)
        self.fc_mu = nn.Linear(self.feature_dim, self.feature_dim)
        self.fc_var = nn.Linear(self.feature_dim, self.feature_dim)

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

        h = conv.reshape(conv.size(0), -1)
        return h

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        h_fc = torch.relu(h_fc)

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
    pretrained_ae = torch.load(args.ae_path, map_location=device)

    encoder = PixelEncoder(env.observation_space.shape, args.ae_dim).to(device)
    encoder.load_state_dict(pretrained_ae['encoder'])
    decoder = PixelDecoder(env.observation_space.shape, args.ae_dim).to(device)
    decoder.load_state_dict(pretrained_ae['decoder'])

    obs, info = env.reset()
    grid = env.unwrapped.grid
    observations = np.zeros(((grid.width - 2) * (grid.height - 2) * 4,) + env.observation_space.shape, dtype=np.uint8)
    labels = ['safe' for _ in range((grid.width - 2) * (grid.height - 2) * 4)]
    names = ['i, j, dir' for _ in range((grid.width - 2) * (grid.height - 2) * 4)]

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
                    assert obs_idx % 4 == dir
                    assert (obs_idx // 4) % (grid.height - 2) == j - 1
                    assert (obs_idx // (4 * (grid.height - 2))) == i - 1
                    next_c = grid.get(*env.unwrapped.front_pos)
                    if c is not None and c.type == 'lava':
                        labels[obs_idx] = 'death'
                    if c is None and next_c is not None and next_c.type == 'lava':
                        labels[obs_idx] = 'unsafe'
                    names[obs_idx] = f'{i}, {j}, {dir}'


    observations = observations.transpose(0, 3, 1, 2)
    observations = np.ascontiguousarray(observations)
    embeddings, mu, log_var = encoder.sample(torch.Tensor(observations).to(device))
    reconstruction = decoder(embeddings)
    assert encoder.outputs['obs'].shape == reconstruction.shape
    reconstruct_loss = torch.nn.functional.mse_loss(reconstruction, encoder.outputs['obs'])
    kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

    print(f'Reconstruction loss: {reconstruct_loss.item()}')
    print(f'KL loss: {kl_loss.item()}')
    X = embeddings.detach().cpu().numpy()

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    print(X_tsne.shape)
    print(tsne.kl_divergence_)
    df = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'])
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
        fig.write_html('tsne_visualization_with_index.html')

        # Show plot in browser
        fig.show()
    else:
        palette = {"safe": "C0", "unsafe": "C1", "death": "C2"}
        ax = sns.scatterplot(data=df, x='TSNE1', y='TSNE2', hue='label', palette=palette)
        plt.title('t-SNE visualization')
        plt.savefig(f't-SNE_{args.env_id}_{args.seed}.png')
        plt.show()
