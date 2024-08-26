from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import random
import minigrid

class QNetwork(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        return self.network(x / 255.0)


def compute_pseudo_reward(Qs, obs, action, next_obs, threshold, gamma):
    # Compute undesirability
    N = len(Qs)
    with torch.no_grad():
        Qvalues = torch.cat([Q(obs) for Q in Qs], dim=0)
        nextQvalues = torch.cat([Q(next_obs) for Q in Qs], dim=0)
        Wsa = torch.abs((Qvalues[:, action] / Qvalues.max(dim=1)[0]) - 1)
        Wsa_probs = F.softmax(Wsa, dim=0)

    # Compute entropy and mean
    entropy = (-Wsa_probs * torch.log(Wsa_probs)).sum() / np.log(N)
    mean = Wsa.mean()
    pseudo_reward = np.array([0.0])
    # Select undesirable actions
    if (mean * entropy).item() > threshold:
        pseudo_reward = Wsa_probs.dot(Qvalues[:, action] - gamma * nextQvalues.max(dim=1)[0]).unsqueeze(0).cpu().numpy()
    return (mean * entropy).item(), pseudo_reward

def visualize_pseudo_reward(q_path, env_id, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    env = gym.make(env_id, render_mode='rgb_array', max_steps=100)
    env = minigrid.wrappers.RGBImgObsWrapper(env)
    env = minigrid.wrappers.ImgObsWrapper(env)
    env = minigrid.wrappers.NoDeath(env, no_death_types=('lava',))
    env = minigrid.wrappers.ReseedWrapper(env, seeds=(seed,))
    q_priors = load_q_priors(env, q_path, device)

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


def load_q_priors(envs, path, device):
    files = Path(path).rglob('*.cleanrl_model')
    Qs = []
    for file in files:
        if isinstance(envs, gym.vector.VectorEnv):
            n_actions = envs.single_action_space.n
        else:
            n_actions = envs.action_space.n

        q_network = QNetwork(n_actions).to(device)
        q_network.load_state_dict(torch.load(str(file), map_location=device))
        Qs.append(q_network)
    return Qs
