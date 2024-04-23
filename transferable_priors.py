import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)

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