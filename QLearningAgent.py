import os
from dataclasses import dataclass

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random
from stable_baselines3.common.buffers import ReplayBuffer


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
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 50000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 5000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""


# ALGO LOGIC: initialize agent here:
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
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class QLearningAgent:
    def __init__(self, env, writer, device=None):
        args = Args()
        self.args = args
        self.env = env
        self.device = device if device is not None \
            else torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.q_network = QNetwork(env).to(device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=args.learning_rate)
        self.target_network = QNetwork(env).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.rb = ReplayBuffer(args.buffer_size, env.observation_space,
                               env.action_space, device, handle_timeout_termination=False)
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
