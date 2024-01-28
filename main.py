import random
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import minigrid
from gymnasium.core import WrapperObsType


class TestWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        print("resetting")
        return self.env.reset(seed=seed, options=options)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    def make_env():
        env = gym.make('MiniGrid-LavaCrossingS9N1-v0', max_steps=512, render_mode='rgb_array')
        env = minigrid.wrappers.RGBImgPartialObsWrapper(env)
        env = minigrid.wrappers.ImgObsWrapper(env)
        env = TestWrapper(env)
        return env

    envs = gym.vector.SyncVectorEnv(
        [make_env for i in range(1)],
    )
    obs, info = envs.reset()
    print(obs.shape)
    print(envs.single_observation_space.shape)
    plt.imshow(obs[0, :, :, :])
    plt.show()
    print(obs[0, :, :, 0])
    for i in range(1000):
        actions = np.random.randint(envs.single_action_space.n, size=1)
        obs, reward, terminated, truncated, info = envs.step(actions)
        if "final_info" in info:
            print(f"{truncated=}, {terminated=}")
