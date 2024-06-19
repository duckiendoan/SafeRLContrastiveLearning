import torch
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


@dataclass
class Args:
    env_id: str = "MiniGrid-LavaCrossingS9N1-v0"
    """the id of the environment"""
    seed: int = 1
    """seed of the experiment"""


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
    obs, info = env.reset()
    grid = env.unwrapped.grid

    X = np.load('runs/MiniGrid-LavaCrossingS9N1-v0_obs_embeddings.npy')
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    print(X_tsne.shape)
    print(tsne.kl_divergence_)
    df = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'])
    labels = ['safe' for _ in range(X_tsne.shape[0])]
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

        if next_c is not None and next_c.type == 'lava':
            labels[obs_idx] = 'unsafe'

    df['labels'] = labels
    sns.scatterplot(data=df, x='TSNE1', y='TSNE2', hue='labels')
    plt.title('t-SNE visualization')
    plt.show()
