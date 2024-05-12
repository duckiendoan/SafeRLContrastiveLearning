import os

import minigrid
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np

from utils import TransposeImageWrapper


class QNetwork(nn.Module):
    def __init__(self, env):
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
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)

class Policy(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.env = env

    def get_action(self, obs):
        return self.env.single_action_space.sample()
def compute_prior(Qs, policy, env, gamma, update_frequency, tau, steps, threshold):
    N = len(Qs)
    Qp = QNetwork(env)
    target_network = QNetwork(env)
    target_network.load_state_dict(Qp.state_dict())

    optimizer = torch.optim.Adam(Qp.parameters(), lr=1e-4)
    obs, info = env.reset()
    for step in range(steps):
        action = policy.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        obs = next_obs
        Wsa = torch.Tensor((N,))
        for i in range(N):
            qi = Qs[i](obs)
            advantage = qi[action] - qi.max(dim=-1)
            Wsa[i] = advantage / qi.max(dim=-1)
        Wsa_probs = F.softmax(Wsa)
        entropy = (-Wsa_probs * torch.log(Wsa_probs)).sum()
        mean = Wsa.mean()
        pseudo_reward = 0
        if mean * entropy > threshold:
            pseudo_reward = Wsa_probs * torch.Tensor([Qs[j](obs)[action] - gamma * Qs[j](next_obs).max(dim=-1)
                                                      for j in range(N)])
        target_reward = pseudo_reward + gamma * target_network(next_obs).max(dim=-1)
        print(target_reward)
        loss = torch.sum((Qp(obs) - target_reward) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % update_frequency == 0:
            for target_network_param, q_network_param in zip(target_network.parameters(), Qp.parameters()):
                target_network_param.data.copy_(
                    tau * q_network_param.data + (1.0 - tau) * target_network_param.data
                )
        if terminated or truncated:
            obs, info = env.reset()
    return Qp

def load_q_priors(env, path):
    files = os.listdir(path)
    Qs = []
    for file in files:
        if file.endswith("cleanrl_model"):
            q_network = QNetwork(env)
            q_network.load_state_dict(torch.load(os.path.join(path, file), map_location=torch.device('cpu')))
            Qs.append(q_network)
    return Qs

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
            env = minigrid.wrappers.ReseedWrapper(env, seeds=(seed,))
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk


if __name__ == "__main__":
    env = gym.vector.SyncVectorEnv([make_env("MiniGrid-LavaCrossingS9N2-v0",
                   10, 0, False,
                   "minigrid_crossing")])

    policy = Policy(env)
    Qs = load_q_priors(env, "./qpriors")
    compute_prior(Qs, policy, env, 0.99, 100, 1.0, 10000, 0.1)