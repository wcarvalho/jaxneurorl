"""

TESTING:
JAX_TRACEBACK_FILTERING=off python -m ipdb -c continue singleagent/baselines.py \
  --debug=True \
  --wandb=False \
  --search=alpha

JAX_DISABLE_JIT=1 JAX_TRACEBACK_FILTERING=off python -m ipdb -c continue singleagent/baselines.py \
  --debug=True \
  --wandb=False \
  --search=alpha

TESTING SLURM LAUNCH:
python singleagent/baselines.py \
  --parallel=sbatch \
  --debug_parallel=True \
  --search=alpha

RUNNING ON SLURM:
python singleagent/baselines.py \
  --parallel=sbatch \
  --time '0-00:30:00' \
  --search=alpha
"""

from absl import flags
from absl import app

import os
import jax
from typing import Dict, Union

from functools import partial
from ray import tune

import wandb
from safetensors.flax import save_file
from flax.traverse_util import flatten_dict

import hydra
import gymnax
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper
from library.wrappers import TimestepWrapper

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import library.flags
from library import parallel
from singleagent import value_based_basics as vbb

from singleagent import alphazero
from singleagent import qlearning
FLAGS = flags.FLAGS

def run_single(
        config: dict,
        save_path: str = None):

    assert config['ENV_NAME'] in (
       'CartPole-v1',
       'Breakout-MinAtar',
       'Catch-bsuite'
    ), 'only these have been tested so far'

    basic_env, env_params = gymnax.make(config['ENV_NAME'])
    env = FlattenObservationWrapper(basic_env)
    # converts to using timestep
    env = TimestepWrapper(env, autoreset=True)
    # env = LogWrapper(env)

    alg_name = config['alg']
    if alg_name == 'qlearning':
      make_train = qlearning.make_train_preloaded
    elif alg_name == 'qlearning_mlp':
      make_train = partial(
        vbb.make_train,
        make_agent=qlearning.make_mlp_agent,
        make_optimizer=qlearning.make_optimizer,
        make_loss_fn_class=qlearning.make_loss_fn_class,
        make_actor=qlearning.make_actor,
      )
    elif alg_name == 'alphazero':
      make_train = alphazero.make_train_preloaded(config)

    else:
      raise NotImplementedError(alg_name)

    train_fn = make_train(config, env, env_params)
    train_vjit = jax.jit(jax.vmap(train_fn))

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    outs = jax.block_until_ready(train_vjit(rngs))

    if save_path is not None:
        def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
            flattened_dict = flatten_dict(params, sep=',')
            save_file(flattened_dict, filename)

        model_state = outs['runner_state'][0]
        params = jax.tree_map(lambda x: x[0], model_state.params) # save only params of the firt run
        os.makedirs(save_path, exist_ok=True)
        save_params(params, f'{save_path}/{alg_name}.safetensors')
        print(f'Parameters of first batch saved in {save_path}/{alg_name}.safetensors')

def sweep(search: str = ''):
  search = search or 'baselines'
  if search == 'baselines':
    space = [
        {
            "group": tune.grid_search(['baselines-Catch-12']),
            "alg": tune.grid_search(['qlearning']),
            "config_name": tune.grid_search(['qlearning']),
            "ENV_NAME": tune.grid_search(['Catch-bsuite']),
        },
        {
            "group": tune.grid_search(['baselines-CartPole-12']),
            "alg": tune.grid_search(['qlearning']),
            "config_name": tune.grid_search(['qlearning']),
            "ENV_NAME": tune.grid_search(['CartPole-v1',]),
        },
        {
            "group": tune.grid_search(['baselines-Breakout-13']),
            "alg": tune.grid_search(['qlearning']),
            "config_name": tune.grid_search(['qlearning']),
            "ENV_NAME": tune.grid_search(['Breakout-MinAtar',]),
        },
    ]
  elif search == 'alpha':
    space = [
        {
            "group": tune.grid_search(['alphazero-Catch-1']),
            "alg": tune.grid_search(['alphazero']),
            "config_name": tune.grid_search(['alphazero']),
            "ENV_NAME": tune.grid_search(['Catch-bsuite']),
        },
        {
            "group": tune.grid_search(['alphazero-CartPole-1']),
            "alg": tune.grid_search(['alphazero']),
            "config_name": tune.grid_search(['alphazero']),
            "ENV_NAME": tune.grid_search(['CartPole-v1',]),
        },
        {
            "group": tune.grid_search(['alphazero-Breakout-1']),
            "alg": tune.grid_search(['alphazero']),
            "config_name": tune.grid_search(['alphazero']),
            "ENV_NAME": tune.grid_search(['Breakout-MinAtar',]),
        },
    ]
  else:
    raise NotImplementedError(search)

  return space

def main(_):
  parallel.run(
      trainer_filename=__file__,
      config_path='configs',
      run_fn=run_single,
      sweep_fn=sweep,
      folder=os.environ.get(
          'RL_RESULTS_DIR', '/tmp/rl_results_dir')
  )

if __name__ == '__main__':
  app.run(main)