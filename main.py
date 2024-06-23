import random
from typing import Any, SupportsFloat

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import minigrid
from gymnasium.core import WrapperObsType, WrapperActType
from stable_baselines3.common.buffers import DictReplayBuffer
from PIL import Image
from utils import SafetyAwareObservationWrapper


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

AGENT_DIRS = {
    0: 'right',
    1: 'down',
    2: 'left',
    3: 'up'
}


def get_action(agent_pos, agent_dir, goalPos):
    dx, dy = goalPos[0] - agent_pos[0], goalPos[1] - agent_pos[1]
    # Left
    if dx < 0:
        if AGENT_DIRS[agent_dir] != 'left':
            return 0
        else:
            return 2
    elif dx > 0:
        if AGENT_DIRS[agent_dir] != 'right':
            return 0
        else:
            return 2

    if dy < 0:
        if AGENT_DIRS[agent_dir] != 'up':
            return 0
        else:
            return 2

    elif dy > 0:
        if AGENT_DIRS[agent_dir] != 'down':
            return 0
        else:
            return 2


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    def make_env():
        env = gym.make('MiniGrid-LavaCrossingS9N1-v0', render_mode='rgb_array')
        env = minigrid.wrappers.RGBImgPartialObsWrapper(env)
        env = minigrid.wrappers.ImgObsWrapper(env)
        env = SafetyAwareObservationWrapper(env)

        return env

    envs = gym.vector.SyncVectorEnv(
        [make_env for i in range(1)],
    )
    rb = DictReplayBuffer(
        20,
        envs.single_observation_space,
        envs.single_action_space,
        handle_timeout_termination=False,
    )
    obs, info = envs.reset()

    print(obs['image'].shape)
    print(envs.single_observation_space['image'])

    # im = Image.fromarray(envs.get_attr("render")[0])
    # im.save("runs/images/1.jpg")

    # plt.imshow(obs[0, :, :, :])
    # plt.show()
    closest_lava = (-1, -1)
    grid = envs.get_attr('grid')[0]
    for i in range(grid.width):
        for j in range(grid.height):
            c = grid.get(i, j)
            if c is not None and c.type == 'lava':
                closest_lava = (i, j)

    for i in range(20):
        # actions = np.random.randint(envs.single_action_space.n, size=1)
        actions = np.array([get_action(envs.get_attr('agent_pos')[0], envs.get_attr('agent_dir')[0], closest_lava)])
        next_obs, reward, terminated, truncated, infos = envs.step(actions)
        if "final_info" in infos:
            for idx, info in enumerate(infos['final_info']):
                print(f"{truncated=}, {terminated=}")
                obs['unsafe'][idx] = info['unsafe']
            # print(obs)
        rb.add(obs, next_obs, actions, reward, truncated, infos)
        obs = next_obs

    print(rb.sample(5).actions.shape)
    print(rb.sample(5).observations['unsafe'].dtype)
    # print(rb.sample(5).observations)

