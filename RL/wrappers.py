import time

import numpy as np
import gymnasium as gym
from gymnasium.wrappers.normalize import NormalizeObservation, NormalizeReward
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics


class MySingleRecordEpisodeStatistics(RecordEpisodeStatistics):
    def __init__(self, env: gym.Env, deque_size: int = 100):
        super().__init__(env, deque_size)

    def step(self, action):
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(action)
        assert isinstance(
            infos, dict
        ), f"`info` dtype is {type(infos)} while supported dtype is `dict`. This may be due to usage of other wrappers in the wrong order."
        self.episode_returns += rewards
        self.episode_lengths += 1
        dones = np.logical_or(terminations, truncations)
        num_dones = np.sum(dones)
        if num_dones:
            if "episode" in infos or "_episode" in infos:
                raise ValueError(
                    "Attempted to add episode stats when they already exist"
                )
            else:
                if hasattr(self, "warmup"):
                    infos["tts"] = self.get_tts(self.warmup, self.total_step)
                else:
                    infos["tts"] = self.get_tts(0, self.total_model_step)
                infos["episode"] = {
                    "r": np.where(dones, self.episode_returns, 0.0),
                    "l": np.where(dones, self.episode_lengths, 0),
                    "t": np.where(
                        dones,
                        np.round(time.perf_counter() - self.episode_start_times, 6),
                        0.0,
                    ),
                }
                if self.is_vector_env:
                    infos["_episode"] = np.where(dones, True, False)
            self.return_queue.extend(self.episode_returns[dones])
            self.length_queue.extend(self.episode_lengths[dones])
            self.episode_count += num_dones
            self.episode_lengths[dones] = 0
            self.episode_returns[dones] = 0.
            self.episode_start_times[dones] = time.perf_counter()
        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )


class RecordNormalParam(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.has_obs_rms = hasattr(env, "obs_rms")  # whether record the ob_mean and ob_var
        self.has_return_rms = hasattr(env, "return_rms")  # whether record the r_mean and r_var

    def step(self, action):
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(action)
        dones = np.logical_or(terminations, truncations)
        num_dones = np.sum(dones)
        if num_dones:
            # record only when the env done
            if self.has_obs_rms and self.has_return_rms:
                infos["ob_mean"], infos["ob_var"] = self.obs_rms.mean, self.obs_rms.var
                infos["r_mean"], infos["r_var"] = self.return_rms.mean, self.return_rms.var
            else:
                infos["ob_mean"], infos["ob_var"] = None, None
                infos["r_mean"], infos["r_var"] = None, None
        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )


class MyNormalizeObservation(NormalizeObservation):
    def __init__(self, env: gym.Env, epsilon: float = 1e-8, count: int = 10000,
                 running_mean: np.ndarray = None, running_var: np.ndarray = None):
        super().__init__(env, epsilon)
        if running_mean is not None and running_var is not None:
            self.obs_rms.mean = running_mean
            self.obs_rms.var = running_var
            self.obs_rms.count = count


class MyNormalizeReward(NormalizeReward):
    def __init__(self, env: gym.Env, gamma: float = 0.99, epsilon: float = 1e-8, count: int = 10000,
                 running_mean: np.ndarray = None, running_var: np.ndarray = None):
        super().__init__(env, gamma, epsilon)
        if running_mean is not None and running_var is not None:
            self.return_rms.mean = running_mean
            self.return_rms.var = running_var
            self.return_rms.count = count
