import random
from typing import Any, SupportsFloat

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import minigrid
from gymnasium.core import WrapperObsType, WrapperActType
from stable_baselines3.common.buffers import ReplayBuffer
from PIL import Image

class TestWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        obs, reward, terminated, truncated, info = super().step(action)
        info['cost'] = 0.0
        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        print("resetting")
        if self.env.has_reset:
            print("Before", self.env.unwrapped.agent_pos, self.env.unwrapped.agent_dir)
            print("continue for 3 more steps")
            self.env.unwrapped.step_count = 0
            for i in range(3):
                obs, reward, terminated, truncated, info = self.env.step(np.random.randint(self.env.action_space.n))
                print(i, self.env.unwrapped.agent_pos, self.env.unwrapped.agent_dir, truncated)
            print("After", self.env.unwrapped.agent_pos, self.env.unwrapped.agent_dir)
        return self.env.reset(seed=seed, options=options)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    def make_env():
        env = gym.make('MiniGrid-LavaCrossingS9N1-v0', max_steps=3, render_mode='rgb_array')
        env = minigrid.wrappers.RGBImgPartialObsWrapper(env)
        env = minigrid.wrappers.ImgObsWrapper(env)
        env = TestWrapper(env)

        return env

    envs = gym.vector.SyncVectorEnv(
        [make_env for i in range(1)],
    )
    rb = ReplayBuffer(
        10,
        envs.single_observation_space,
        envs.single_action_space,
        handle_timeout_termination=False,
    )
    obs, info = envs.reset()

    print(obs.shape)
    print(envs.single_observation_space.shape)

    # im = Image.fromarray(envs.get_attr("render")[0])
    # im.save("runs/images/1.jpg")

    # plt.imshow(obs[0, :, :, :])
    # plt.show()
    for i in range(10):
        print(i)
        actions = np.random.randint(envs.single_action_space.n, size=1)
        next_obs, reward, terminated, truncated, info = envs.step(actions)
        rb.add(obs, next_obs, actions, reward, truncated, info)
        if "final_info" in info:
            print(f"{truncated=}, {terminated=}")
        obs = next_obs

    print(rb.sample(5).actions.shape)