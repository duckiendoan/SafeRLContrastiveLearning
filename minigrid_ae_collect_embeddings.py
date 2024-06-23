import gymnasium as gym
import minigrid
from dataclasses import dataclass
import numpy as np
import random
import tyro
import torch
import torch.nn as nn

@dataclass
class Args:
    encoder_path: str = './encoder.pth'
    """path to encoder"""
    decoder_path: str = './decoder.pth'
    """path to decoder"""
    env_id: str = "MiniGrid-LavaCrossingS9N1-v0"
    """the id of the environment"""
    seed: int = 1
    """seed of the experiment"""
    ae_dim: int = 50
    """the dimensionality of the latent space"""

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


AGENT_DIRS = {
    0: 'right',
    1: 'down',
    2: 'left',
    3: 'up'
}


def get_action(agent_pos, agent_dir, goalPos):
    dx, dy = goalPos[0] - agent_pos[0], goalPos[1] - agent_pos[1]
    # Left
    if dx < 0:
        if AGENT_DIRS[agent_dir] != 'left':
            return 0
        else:
            return 2
    elif dx > 0:
        if AGENT_DIRS[agent_dir] != 'right':
            return 0
        else:
            return 2

    if dy < 0:
        if AGENT_DIRS[agent_dir] != 'up':
            return 0
        else:
            return 2

    elif dy > 0:
        if AGENT_DIRS[agent_dir] != 'down':
            return 0
        else:
            return 2


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

    encoder = PixelEncoder(env.observation_space.shape, args.ae_dim).to(device)
    encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))
    decoder = PixelDecoder(env.observation_space.shape, args.ae_dim).to(device)
    decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))

    obs, info = env.reset()
    grid = env.unwrapped.grid
    observations = np.zeros(((grid.width - 2) * (grid.height - 2) * 4,) + env.observation_space.shape, dtype=np.uint8)
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

    observations = observations.transpose(0, 3, 1, 2)
    observations = np.ascontiguousarray(observations)
    embeddings = encoder(torch.Tensor(observations).to(device))
    reconstruction = decoder(embeddings)
    assert encoder.outputs['obs'].shape == reconstruction.shape
    reconstruct_loss = torch.nn.functional.mse_loss(reconstruction, encoder.outputs['obs'])

    print(f'Reconstruction loss: {reconstruct_loss.item()}')
    cpu_embeddings = embeddings.detach().cpu().numpy()
    np.save(f'{args.env_id}__{args.seed}__obs_embeddings.npy', cpu_embeddings)
