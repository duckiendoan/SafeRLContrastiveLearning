import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random
from stable_baselines3.common.buffers import ReplayBuffer

# ALGO LOGIC: initialize agent here:
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

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class QLearningAgent:
    def __init__(self, env, args, writer, device=None):
        self.args = args
        self.env = env
        self.device = device if device is not None \
            else torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.q_network = QNetwork(env).to(device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=args.learning_rate)
        self.target_network = QNetwork(env).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.rb = ReplayBuffer(args.buffer_size, env.single_observation_space,
                               env.single_action_space, device, handle_timeout_termination=False)
        self.global_step = 0
        self.writer = writer

    def choose_action(self, obs):
        args = self.args
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, self.global_step)
        if random.random() < epsilon:
            actions = np.array([self.env.single_action_space.sample() for _ in range(self.env.num_envs)])
        else:
            q_values = self.q_network(torch.Tensor(obs).to(self.device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        return actions

    def get_q(self, obs, action):
        with torch.no_grad():
            q_values = self.q_network(torch.Tensor(obs).to(self.device))
            return torch.squeeze(q_values[:, action].cpu()).item()

    def train(self):
        args = self.args
        if self.global_step > args.learning_starts:
            if self.global_step % args.train_frequency == 0:
                data = self.rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = self.target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = self.q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if self.global_step % 100 == 0:
                    self.writer.add_scalar("reset_losses/td_loss", loss, self.global_step)
                    self.writer.add_scalar("reset_losses/q_values", old_val.mean().item(), self.global_step)

                # optimize the model
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # update target network
            if self.global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )
