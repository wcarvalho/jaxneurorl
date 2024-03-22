"""

TESTING:
JAX_TRACEBACK_FILTERING=off python -m ipdb -c continue singleagent/cartpole.py \
  --debug=True \
  --wandb=False \
  --search=default

JAX_DISABLE_JIT=1 JAX_TRACEBACK_FILTERING=off python -m ipdb -c continue singleagent/cartpole.py \
  --debug=True \
  --wandb=False \
  --search=default

TESTING SLURM LAUNCH:
python singleagent/cartpole.py \
  --parallel=sbatch \
  --debug_parallel=True \
  --search=default

RUNNING ON SLURM:
python singleagent/cartpole.py \
  --parallel=sbatch \
  --search=default
"""

from absl import flags
from absl import app

import os
import jax
from typing import Dict, Union

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
from singleagent import qlearning
FLAGS = flags.FLAGS

def run_single(
        sweep_config: dict,
        config_path: str,
        save_path: str = None):

    config, wandb_init = parallel.load_hydra_config(
        sweep_config,
        config_path=config_path,
        save_path=save_path,
        tags=[f"jax_{jax.__version__}"]
        )
    wandb.init(**wandb_init)

    basic_env, env_params = gymnax.make("CartPole-v1")
    env = FlattenObservationWrapper(basic_env)
    # converts to using timestep
    env = TimestepWrapper(env, autoreset=True)
    # env = LogWrapper(env)

    alg_name = config['alg']
    if alg_name == 'qlearning':
      make_train = qlearning.make_train_preloaded
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
  search = search or 'default'
  if search == 'default':
    space = [
        {
            "group": tune.grid_search(['run-5-qlearning']),
            "alg": tune.grid_search(['qlearning']),
            "AGENT_HIDDEN_DIM": tune.grid_search([64, 128]),
            "AGENT_INIT_SCALE": tune.grid_search([2., .1]),
        }
    ]
  else:
    raise NotImplementedError(search)

  return space

def main(_):
  parallel.run(
      trainer_filename=__file__,
      config_path='../configs',  # must be relative...
      run_fn=run_single,
      sweep_fn=sweep,
      folder=os.environ.get(
          'RL_RESULTS_DIR', '/tmp/rl_results_dir')
  )

if __name__ == '__main__':
  app.run(main)