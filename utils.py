import minigrid
import gymnasium as gym
from gymnasium import Wrapper
from gymnasium.core import WrapperObsType, WrapperActType
from typing import Any, SupportsFloat
from minigrid.core.world_object import Lava
import numpy as np

class TransposeImageWrapper(gym.ObservationWrapper):
    '''Transpose img dimension before being fed to neural net'''
    def __init__(self, env, op=[2,0,1]):
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
    def __init__(self, env, image_path="runs/images"):
        """A wrapper to prevent death in specific cells.

        Args:
            env: The environment to apply the wrapper
            no_death_types: List of strings to identify death cells
            death_cost: The negative reward received in death cells

        """
        super().__init__(env)
        self.env = env
        self.count = 0
        self.image_path = image_path

    def step(self, action):
        # In Dynamic-Obstacles, obstacles move after the agent moves,
        # so we need to check for collision before self.env.step()
        front_cell = self.grid.get(*self.front_pos)
        going_to_death = (
            action == self.actions.forward
            and front_cell is not None
            and front_cell.type in self.no_death_types
        )

        obs, reward, terminated, truncated, info = self.env.step(action)

        # We also check if the agent stays in death cells (e.g., lava)
        # without moving
        current_cell = self.grid.get(*self.agent_pos)
        in_death = current_cell is not None and current_cell.type in self.no_death_types

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
