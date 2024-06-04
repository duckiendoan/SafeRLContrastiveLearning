from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    pseudo_reward = 0.0
    # Select undesirable actions
    if (mean * entropy).item() > threshold:
        pseudo_reward = Wsa_probs.dot(Qvalues[:, action] - gamma * nextQvalues.max(dim=1)[0])
    return (mean * entropy).item(), pseudo_reward


def load_q_priors(envs, path, device):
    files = Path(path).rglob('*.cleanrl_model')
    Qs = []
    for file in files:
        q_network = QNetwork(envs)
        q_network.load_state_dict(torch.load(str(file), map_location=device))
        Qs.append(q_network)
    return Qs
