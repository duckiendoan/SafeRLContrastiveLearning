import random
from typing import Any, SupportsFloat

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import minigrid
from gymnasium.core import WrapperObsType, WrapperActType
from PIL import Image
import safety_gymnasium

class TestWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    def make_env():
        env = gym.make('SafetyPointGoal2Gymnasium-v0', render_mode='rgb_array')
        env = TestWrapper(env)

        return env

    envs = gym.vector.SyncVectorEnv(
        [make_env for i in range(2)],
    )
    obs, info = envs.reset()
    print(obs.shape)
    print(envs.single_observation_space.shape)

    for i in range(10):
        print(i)
        actions = envs.action_space.sample()
        print(actions.shape)
        obs, reward, terminated, truncated, info = envs.step(actions)
        print(info['cost'])
        print(reward)
        if "final_info" in info:
            print(f"{truncated=}, {terminated=}")
