from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import Wrapper
from gymnasium.core import WrapperObsType
from minigrid.core.world_object import Lava


class TransposeImageWrapper(gym.ObservationWrapper):
    '''Transpose img dimension before being fed to neural net'''

    def __init__(self, env, op=[2, 0, 1]):
        super().__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


class DeathLogWrapper(Wrapper):
    def __init__(self, env):
        """A wrapper to prevent death in specific cells.

        Args:
            env: The environment to apply the wrapper
            no_death_types: List of strings to identify death cells
            death_cost: The negative reward received in death cells

        """
        super().__init__(env)

    def step(self, action):
        # In Dynamic-Obstacles, obstacles move after the agent moves,
        # so we need to check for collision before self.env.step()
        front_cell = self.grid.get(*self.front_pos)
        going_to_death = (
                action == self.actions.forward
                and front_cell is not None
                and front_cell.type == 'lava'
        )

        obs, reward, terminated, truncated, info = self.env.step(action)

        # We also check if the agent stays in death cells (e.g., lava)
        # without moving
        current_cell = self.grid.get(*self.agent_pos)
        in_death = current_cell is not None and current_cell.type == 'lava'

        if terminated and (going_to_death or in_death):
            self.count += 1

        return obs, reward, terminated, truncated, info


DIR_MAP = {
    0: np.array([1, 0]),
    1: np.array([0, 1]),
    2: np.array([-1, 0]),
    3: np.array([0, -1])
}


class NearestHazardCostWrapper(gym.Wrapper):
    def __init__(self, env, cost_fn):
        super().__init__(env)
        self.lava_locations = None
        self.cost_fn = cost_fn

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pos = np.array(self.env.unwrapped.agent_pos)
        dir = self.env.unwrapped.agent_dir
        min_d = min(
            [np.sqrt(((pos - lava) ** 2).sum()) for lava in self.lava_locations if (lava - pos).dot(DIR_MAP[dir]) >= 0],
            default=1e6)
        info['cost'] = self.cost_fn(min_d)
        return obs, reward, terminated, truncated, info

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        grid = self.env.unwrapped.grid
        self.lava_locations = [np.array((x % grid.width, x // grid.width)) for x, y in enumerate(grid.grid) if
                               isinstance(y, Lava)]
        return obs, info


class StateCountRecorder:
    """Record state distributions, for ploting visitation frequency heatmap"""

    def __init__(self, env):
        self.shape = env.grid.height, env.grid.width
        self.count = np.zeros(self.shape, dtype=np.int32)
        self.rewards = np.zeros(self.shape, dtype=float)
        self.extract_mask(env)

    def add_count(self, w, h):
        self.count[h, w] += 1

    def add_count_from_env(self, env):
        self.add_count(*env.front_pos)

    def add_reward(self, w, h, r):
        self.rewards[h, w] += r

    def add_reward_from_env(self, env, reward):
        self.add_reward(*env.agent_pos, reward)

    def get_figure_log_scale(self, cap_threshold_cnt=10_000):
        """ plot heat map visitation, similar to `get_figure` but on log scale"""
        import matplotlib
        import matplotlib.pyplot as plt
        cnt = np.clip(self.count + 1, 0, cap_threshold_cnt)
        plt.clf()
        plt.jet()
        plt.imshow(cnt, cmap="jet",
                   norm=matplotlib.colors.LogNorm(vmin=1, vmax=cap_threshold_cnt, clip=True))
        cbar = plt.colorbar()
        cbar.set_label('Visitation counts')

        # over lay walls
        plt.imshow(np.zeros_like(cnt, dtype=np.uint8),
                   cmap="gray", alpha=self.mask.astype(np.float32),
                   vmin=0, vmax=1)
        return plt.gcf()

    def get_figure(self, cap_threshold_cnt=5000):
        import matplotlib.pyplot as plt
        cnt = np.clip(self.count, 0, cap_threshold_cnt)
        plt.clf()
        plt.jet()
        plt.imshow(cnt, cmap="jet", vmin=0, vmax=cap_threshold_cnt)
        cbar = plt.colorbar()
        lin_spc = np.linspace(0, cap_threshold_cnt, 6).astype(np.int32)
        # cbar.ax.yaxis.set_major_locator(ticker.FixedLocator(lin_spc))
        cbar.update_ticks()
        lin_spc = [str(i) for i in lin_spc]
        lin_spc[-1] = ">" + lin_spc[-1]
        cbar.ax.set_yticklabels(lin_spc)
        cbar.set_label('Visitation counts')

        # over lay walls
        plt.imshow(np.zeros_like(cnt, dtype=np.uint8),
                   cmap="gray", alpha=self.mask.astype(np.float32),
                   vmin=0, vmax=1)
        return plt.gcf()

    def extract_mask(self, env):
        """ Extract walls from grid_env, used for masking wall cells in heatmap """
        self.mask = np.zeros_like(self.count)
        for i in range(env.grid.height):
            for j in range(env.grid.width):
                c = env.grid.get(i, j)
                if c is not None and c.type == "wall":
                    self.mask[i, j] = 1

    def save_to(self, file_path):
        with open(file_path, 'wb') as f:
            np.save(f, self.count)
            np.save(f, self.mask)

    def load_from(self, file_path):
        with open(file_path, 'rb') as f:
            self.count = np.load(f)
            self.mask = np.load(f)
        self.shape = self.count.shape
