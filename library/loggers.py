
from typing import Callable, Any, Optional


from flax import struct
from flax.training.train_state import TrainState
from gymnax.environments import environment
import jax
import jax.numpy as jnp
import tree
import wandb

from library.observers import BasicObserverState

def default_gradient_logger(
    train_state: TrainState,
    gradients: dict,
    key: str = ' gradients'):

    subtree_mean = lambda t: jnp.array(tree.flatten(t)).mean()
    subtree_min = lambda t: jnp.array(tree.flatten(t)).min()

    gradients = gradients['params']
    gradients = jax.tree_map(lambda x: x.mean(), gradients)
    gradient_metrics = {
        f'{key}/{k}_mean': subtree_mean(v) for k, v in gradients.items()}
    gradient_metrics.update(
       {f'{key}/{k}_min': subtree_min(v) for k, v in gradients.items()})

    def callback(g):
        import ipdb; ipdb.set_trace()
        if wandb.run is not None:
          wandb.log(g)

    jax.debug.callback(callback, gradient_metrics)

def default_learner_logger(
        train_state: TrainState,
        learner_metrics: dict,
        key: str = 'learner'):

    def callback(ts: train_state, metrics: dict):
        if wandb.run is not None:
          metrics = {f'{key}/{k}': v for k, v in metrics.items()}
          # extra
          extra = {
              'num_actor_steps': train_state.timesteps,
              'num_learner_updates': train_state.n_updates,
          }
          metrics.update(
              {f'{key}/{k}': v for k, v in extra.items()})
          wandb.log(metrics)

    jax.debug.callback(callback, train_state, learner_metrics)


def default_experience_logger(
        train_state: TrainState,
        observer_state: BasicObserverState,
        key: str = 'train'):

    def callback(ts: train_state, os: BasicObserverState):
        if wandb.run is not None:

          # main
          idx = min(os.idx + 1, len(os.episode_lengths))
          metrics = {
              f'{key}/avg_episode_length': os.episode_lengths[:idx].mean(),
              f'{key}/avg_episode_return': os.episode_returns[:idx].mean(),
          }

          # extra
          extra = {
              'num_actor_steps': train_state.timesteps,
              'num_learner_updates': train_state.n_updates,
          }
          metrics.update(
              {f'{key}/{k}': v for k, v in extra.items()})
          wandb.log(metrics)

    jax.debug.callback(callback, train_state, observer_state)


@struct.dataclass
class Logger:

  gradient_logger: Callable[
     [TrainState, dict, str], Any] 

  learner_logger: Callable[
     [TrainState, dict, str], Any]

  experience_logger: Callable[[
      TrainState, BasicObserverState, str], Any]

  learner_log_extra: Optional[Callable[[Any], Any]] = None


def default_make_logger(config: dict,
                        env: environment.Environment,
                        env_params: environment.EnvParams):

    return Logger(
        gradient_logger=default_gradient_logger,
        learner_logger=default_learner_logger,
        experience_logger=default_experience_logger,
   )
