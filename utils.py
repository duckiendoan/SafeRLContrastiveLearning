import minigrid
import gymnasium as gym


class ImgObsWrapper(minigrid.wrappers.ObservationWrapper):
    def __init__(self, env):
        """A wrapper that makes image the only observation.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        img_space = env.observation_space.spaces["image"]
        self.observation_space = gym.spaces.Box(
            low=img_space.low[0, 0, 0],
            high=img_space.high[0, 0, 0],
            shape=(img_space.shape[-1], img_space.shape[-3], img_space.shape[-2]),
            dtype="uint8",
        )

    def observation(self, obs):
        # Transpose observation to match torch's nn
        return obs["image"].swapaxes(0, 2).swapaxes(1, 2)
