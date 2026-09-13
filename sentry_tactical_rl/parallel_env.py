"""Fork-based parallel tactical environments with shared observation buffers."""
from __future__ import annotations

import ctypes
import multiprocessing as mp
from multiprocessing.connection import Connection
import traceback
from typing import Any

import numpy as np

from .env import SentryTacticalEnv


Observation = dict[str, np.ndarray]
_OBSERVATION_KEYS = ("map", "vector", "goal_mask", "target_mask")


def _raw_buffer(ctx: mp.context.BaseContext, dtype: np.dtype[Any], size: int) -> Any:
    if dtype == np.dtype(np.float32):
        return ctx.RawArray(ctypes.c_float, size)
    if dtype == np.dtype(np.bool_):
        return ctx.RawArray(ctypes.c_uint8, size)
    raise TypeError(f"unsupported shared observation dtype: {dtype}")


def _views(
    buffers: dict[str, Any],
    specs: dict[str, tuple[tuple[int, ...], str]],
    num_envs: int,
) -> Observation:
    return {
        key: np.frombuffer(buffers[key], dtype=np.dtype(dtype_name)).reshape((num_envs, *shape))
        for key, (shape, dtype_name) in specs.items()
    }


def _write_observation(views: Observation, index: int, observation: Observation) -> None:
    for key in _OBSERVATION_KEYS:
        np.copyto(views[key][index], observation[key], casting="no")


def _worker(
    connection: Connection,
    env: SentryTacticalEnv,
    index: int,
    seed: int,
    buffers: dict[str, Any],
    specs: dict[str, tuple[tuple[int, ...], str]],
    num_envs: int,
    torch_threads: int,
) -> None:
    try:
        # Frozen sparring inference is small. One Torch thread per worker keeps
        # four environments from oversubscribing all host cores.
        import torch

        torch.set_num_threads(max(1, int(torch_threads)))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        views = _views(buffers, specs, num_envs)
        _write_observation(views, index, env.reset(seed=seed))
        connection.send(("ready", None))
        while True:
            command, payload = connection.recv()
            if command == "step":
                observation, reward, done, info = env.step(payload)
                if done:
                    observation = env.reset()
                _write_observation(views, index, observation)
                connection.send(("ok", (float(reward), bool(done), info)))
            elif command == "reset":
                _write_observation(views, index, env.reset(seed=payload))
                connection.send(("ok", None))
            elif command == "close":
                connection.send(("closed", None))
                break
            else:
                raise ValueError(f"unknown worker command: {command!r}")
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        connection.close()


class ParallelSentryEnvPool:
    """Run independent tactical matches in forked worker processes.

    The environment and its frozen offline policies are loaded once before
    forking, so model pages remain copy-on-write shared. Large raster
    observations use inherited shared buffers instead of being pickled through
    a pipe every tactical second.
    """

    def __init__(
        self,
        prototype_env: SentryTacticalEnv,
        num_envs: int,
        *,
        base_seed: int,
        worker_torch_threads: int = 1,
    ) -> None:
        if num_envs < 2:
            raise ValueError("ParallelSentryEnvPool requires num_envs >= 2")
        available = mp.get_all_start_methods()
        if "fork" not in available:
            raise RuntimeError("parallel tactical training currently requires the Linux 'fork' start method")
        self.num_envs = int(num_envs)
        self.base_seed = int(base_seed)
        self._ctx = mp.get_context("fork")
        example = prototype_env.reset(seed=self.base_seed)
        self._specs = {
            key: (tuple(example[key].shape), np.dtype(example[key].dtype).str)
            for key in _OBSERVATION_KEYS
        }
        self._buffers = {
            key: _raw_buffer(
                self._ctx,
                np.dtype(dtype_name),
                self.num_envs * int(np.prod(shape, dtype=np.int64)),
            )
            for key, (shape, dtype_name) in self._specs.items()
        }
        self._observations = _views(self._buffers, self._specs, self.num_envs)
        self._connections: list[Connection] = []
        self._processes: list[mp.Process] = []
        self._closed = False
        for index in range(self.num_envs):
            parent, child = self._ctx.Pipe()
            process = self._ctx.Process(
                target=_worker,
                args=(
                    child,
                    prototype_env,
                    index,
                    self._seed_for(index),
                    self._buffers,
                    self._specs,
                    self.num_envs,
                    worker_torch_threads,
                ),
                name=f"sentry-env-{index}",
                daemon=True,
            )
            process.start()
            child.close()
            self._connections.append(parent)
            self._processes.append(process)
        try:
            for connection in self._connections:
                self._expect(connection, "ready")
        except BaseException:
            self.close()
            raise

    def _seed_for(self, index: int) -> int:
        # A wide deterministic stride avoids adjacent generators receiving
        # nearly identical match randomisation while keeping runs repeatable.
        return self.base_seed + index * 10_007

    @staticmethod
    def _expect(connection: Connection, expected: str) -> Any:
        status, payload = connection.recv()
        if status == "error":
            raise RuntimeError(f"parallel tactical environment failed:\n{payload}")
        if status != expected:
            raise RuntimeError(f"parallel tactical environment returned {status!r}, expected {expected!r}")
        return payload

    def observations(self, *, copy: bool = False) -> Observation:
        if self._closed:
            raise RuntimeError("parallel environment pool is closed")
        if copy:
            return {key: value.copy() for key, value in self._observations.items()}
        return self._observations

    def reset(self) -> Observation:
        for index, connection in enumerate(self._connections):
            connection.send(("reset", self._seed_for(index)))
        for connection in self._connections:
            self._expect(connection, "ok")
        return self.observations()

    def step(self, actions: np.ndarray) -> tuple[Observation, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.num_envs, 3):
            raise ValueError(f"actions must have shape ({self.num_envs}, 3), got {actions.shape}")
        for connection, action in zip(self._connections, actions):
            connection.send(("step", action))
        results = [self._expect(connection, "ok") for connection in self._connections]
        rewards = np.asarray([result[0] for result in results], dtype=np.float32)
        dones = np.asarray([result[1] for result in results], dtype=np.bool_)
        infos = [result[2] for result in results]
        return self.observations(), rewards, dones, infos

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for connection, process in zip(self._connections, self._processes):
            if process.is_alive():
                try:
                    connection.send(("close", None))
                except (BrokenPipeError, EOFError):
                    pass
        for connection, process in zip(self._connections, self._processes):
            if process.is_alive():
                try:
                    self._expect(connection, "closed")
                except (BrokenPipeError, EOFError, RuntimeError):
                    pass
            connection.close()
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)

    def __enter__(self) -> "ParallelSentryEnvPool":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown fallback
        try:
            self.close()
        except Exception:
            pass
