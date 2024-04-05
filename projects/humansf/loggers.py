
import abc
import collections
from typing import Dict, Union, Optional, Callable

import jax
import jax.numpy as jnp
import flashbax as fbx
from flax.training.train_state import TrainState
from flax import struct
import numpy as np
import wandb

from singleagent.basics import TimeStep

Number = Union[int, float, np.float32, jnp.float32]

def default_experience_logger(
        train_state: TrainState,
        observer_state: BasicObserverState,
        key: str = 'train'):

    def callback(ts: train_state, os: BasicObserverState):
        # main
        end = min(os.idx + 1, len(os.episode_lengths))
        metrics = {
            f'{key}/avg_episode_length': os.episode_lengths[:end].mean(),
            f'{key}/avg_episode_return': os.episode_returns[:end].mean(),
            f'{key}/num_actor_steps': ts.timesteps,
            f'{key}/num_learner_updates': ts.n_updates,
        }
        if wandb.run is not None:
          wandb.log(metrics)

    jax.debug.callback(callback, train_state, observer_state)
