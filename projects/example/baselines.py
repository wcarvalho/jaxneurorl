"""

TESTING:
JAX_TRACEBACK_FILTERING=off python -m ipdb -c continue agents/baselines.py \
  --debug=False \
  --wandb=False \
  --search=alpha

JAX_DISABLE_JIT=1 JAX_TRACEBACK_FILTERING=off python -m ipdb -c continue agents/baselines.py \
  --debug=True \
  --wandb=False \
  --search=alpha

TESTING SLURM LAUNCH:
python agents/baselines.py \
  --parallel=sbatch \
  --debug_parallel=True \
  --search=alpha

RUNNING ON SLURM:
python agents/baselines.py \
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


from safetensors.flax import save_file
from flax.traverse_util import flatten_dict


import gymnax
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper
from jaxneurorl.wrappers import TimestepWrapper

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

from jaxneurorl import launcher
from jaxneurorl.agents import value_based_basics as vbb

from jaxneurorl.agents import alphazero
from jaxneurorl.agents import qlearning

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

    alg_name = config['alg']
    if alg_name == 'qlearning':
      make_train = qlearning.make_train_preloaded
    elif alg_name == 'qlearning_mlp':
      make_train = partial(
        vbb.make_train,
        make_agent=qlearning.make_mlp_agent,  # only difference from above
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
  if search == 'qlearning':
    sweep_config = {
        'metric': {
            'name': 'evaluator_performance/0.0 avg_episode_return',
            'goal': 'maximize',
        },
        'parameters': {
            "ENV_NAME": {'values:' ['Catch-bsuite']},
        },
        'overrides': ['alg=qlearning',
                      'rlenv=cartpole',
                      'user=wilka'],
        'group': 'qlearning-1',
    }
  elif search == 'pqn':
    sweep_config = {
        'metric': {
            'name': 'evaluator_performance/0.0 avg_episode_return',
            'goal': 'maximize',
        },
        'parameters': {
            "env.exp": {'values': [
                'maze3_open',
            ]},
            "BATCH_SIZE": {'values': [512*128, 256*128, 128*128]},
            "NORM_TYPE": {'values': ['layer_norm', 'none']},
            "NORM_QFN": {'values': ['layer_norm', 'none']},
            "TOTAL_TIMESTEPS": {'values': [100_000_000]},
        },
        'overrides': ['alg=pqn', 'rlenv=cartpole', 'user=wilka'],
        'group': 'pqn-1',
    }
  elif search == 'alpha':
    sweep_config = {
        'metric': {
            'name': 'evaluator_performance/0.0 avg_episode_return',
            'goal': 'maximize',
        },
        'parameters': {
            "ENV_NAME": {'values:' ['Catch-bsuite']},
        },
        'overrides': ['alg=alphazero',
                      'rlenv=cartpole',
                      'user=wilka'],
        'group': 'alpha-1',
    }
  else:
    raise NotImplementedError(search)

  return sweep_config

if __name__ == '__main__':
  from jaxneurorl.utils import make_parser
  parser = make_parser()
  args = parser.parse_args()
  launcher.run(
      args,
      trainer_filename=__file__,
      config_path='configs',
      run_fn=run_single,
      sweep_fn=sweep,
      folder=os.environ.get(
          'RL_RESULTS_DIR', '/tmp/rl_results_dir')
  )