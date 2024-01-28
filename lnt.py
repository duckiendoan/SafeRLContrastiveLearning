# Adapted from https://github.com/brain-research/LeaveNoTrace/blob/master/lnt.py
from typing import Any

import gymnasium as gym
from gymnasium import Wrapper
import numpy as np


class SafetyWrapper(Wrapper):
    def __init__(self, env, reset_agent, reset_reward_fn, reset_done_fn, q_min):
        super().__init__(env)
        self._reset_agent = reset_agent
        self._reset_reward_fn = reset_reward_fn
        self._reset_done_fn = reset_done_fn
        self._q_min = q_min
        # Get max_steps for Minigrid environments
        self._max_steps = env.get_wrapper_attr("max_steps") // 2
        self.obs, _ = env.reset()

        # Setup internal structures for logging metrics.
        self._total_resets = 0  # Total resets taken during training
        self._episode_rewards = []  # Rewards for the current episode
        self._reset_history = []
        self._reward_history = []

    def _reset(self, seed, options):
        """
        Run the reset policy
        :return: obs, reward, terminated, truncated, info
        """
        obs = self.obs
        for t in range(self._max_steps):
            self._reset_agent.global_step += 1
            reset_action = self._reset_agent.choose_action(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action=reset_action)
            # reset_reward = self._reset_reward_fn(next_obs, reset_action)
            # reset_done = self._reset_done_fn(next_obs)
            # For Minigrid
            reset_done = np.array_equal(self.env.unwrapped.agent_pos, np.array([1, 1]))
            reset_reward = float(reset_done) - 1.0
            self._reset_agent.rb.add(obs[np.newaxis], next_obs[np.newaxis], reset_action, reset_reward, reset_done, info)
            obs = next_obs
            self._reset_agent.train()
            if reset_done:
                break

        # Fail to reset after time limit
        if not reset_done:
            obs, info = self.env.reset(seed=seed, options=options)
            self._total_resets += 1
        # Log metrics
        self._reset_history.append(self._total_resets)
        self._reward_history.append(np.mean(self._episode_rewards))
        self._episode_rewards = []

        # If the agent takes an action that causes an early abort the agent
        # shouldn't believe that the episode terminates. Because the reward is
        # negative, the agent would be incentivized to do early aborts as
        # quickly as possible. Thus, we set done = False.
        done = False

        # Reset the elapsed steps back to 0
        self.env._elapsed_steps = 0
        return obs, reward, done, False, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, reward, terminated, truncated, info = self._reset(seed, options)
        return obs, info

    def step(self, action):
        # Calculate the Q value when in current state and taking an action
        reset_q = self._reset_agent.get_q(self.obs, action)
        # If the action is unsafe, run the reset policy
        if reset_q < self._q_min:
            obs, reward, terminated, truncated, info = self._reset(seed=None, options=None)
        else:
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._episode_rewards.append(reward)
        self.obs = obs
        return obs, reward, terminated, truncated, info

    def plot_metrics(self, output_dir='/tmp'):
        """
        Plot metrics collected during training.

        args:
            output_dir: (optional) folder path for saving results.
        """

        import matplotlib.pyplot as plt
        import json
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        data = {
            'reward_history': self._reward_history,
            'reset_history': self._reset_history
        }
        with open(os.path.join(output_dir, 'data.json'), 'w') as f:
            json.dump(data, f)

        # Prepare data for plotting
        rewards = np.array(self._reward_history)
        lnt_resets = np.array(self._reset_history)
        num_episodes = len(rewards)
        baseline_resets = np.arange(num_episodes)
        episodes = np.arange(num_episodes)

        # Plot the data
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.gca()
        ax2 = ax1.twinx()
        ax1.plot(episodes, rewards, 'g.')
        ax2.plot(episodes, lnt_resets, 'b-')
        ax2.plot(episodes, baseline_resets, 'b--')

        # Label the plot
        ax1.set_ylabel('average step reward', color='g', fontsize=20)
        ax1.tick_params('y', colors='g')
        ax2.set_ylabel('num. resets', color='b', fontsize=20)
        ax2.tick_params('y', colors='b')
        ax1.set_xlabel('num. episodes', fontsize=20)
        plt.savefig(os.path.join(output_dir, 'plot.png'))

        plt.show()
