"""

TESTING:
JAX_DISABLE_JIT=1 \
HYDRA_FULL_ERROR=1 JAX_TRACEBACK_FILTERING=off python -m ipdb -c continue projects/example/baselines.py \
  app.debug=True \
  app.wandb=False \
  app.search=pqn

RUNNING LOCAL WANDB SWEEP:
python projects/example/baselines.py \
  app.parallel=wandb \
  app.search=pqn

RUNNING WANDB SWEEP ON SLURM:
python projects/example/baselines.py \
  app.parallel=sbatch \
  app.parent=wandb_search \
  app.search=pqn

"""

import hydra
from omegaconf import DictConfig


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
from jaxneurorl.agents import value_based_pqn as vpq

from jaxneurorl.agents import alphazero
from jaxneurorl.agents import qlearning
from jaxneurorl.agents import dyna
from projects.example import housemaze_dyna

gymnax_envs = ['CartPole-v1',
       'Breakout-MinAtar',
       'Catch-bsuite']

def run_single(
        config: dict,
        save_path: str = None):

    assert config['ENV_NAME'] in (
       'CartPole-v1',
       'Breakout-MinAtar',
       'Catch-bsuite',
       'housemaze'
    ), 'only these have been tested so far'

    if config['ENV_NAME'] in gymnax_envs:
      basic_env, env_params = gymnax.make(config['ENV_NAME'])
    elif config['ENV_NAME'] == "housemaze":
       env_params, basic_env = housemaze_dyna.housemaze_env()

    env = basic_env
    if config['ENV_NAME'] != "housemaze":
      env = FlattenObservationWrapper(basic_env)

      # converts to using timestep
      env = TimestepWrapper(env, autoreset=True)

    alg_name = config['ALG']
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
    elif alg_name == 'pqn':
      make_train = vpq.make_train
    elif alg_name == 'alphazero':
      make_train = alphazero.make_train_preloaded(config)
    elif alg_name == 'qlearning_dyna':
      make_train = dyna.make_train_preloaded

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
  search = search or 'qlearning'
  if search == 'qlearning':
    sweep_config = {
        'metric': {
            'name': 'evaluator_performance/0.0 avg_episode_return',
            'goal': 'maximize',
        },
        'parameters': {
            "ENV_NAME": {'values': ['Catch-bsuite']},
            "AGENT_HIDDEN_DIM": {'values': [32, 64, 128, 256]},
        },
        'overrides': ['alg=qlearning'],
        'group': 'qlearning-4',
    }
  elif search == 'pqn':
    sweep_config = {
        'metric': {
            'name': 'evaluator_performance/0.0 avg_episode_return',
            'goal': 'maximize',
        },
        'parameters': {
            "ENV_NAME": {'values': ['Catch-bsuite']},
            "BATCH_SIZE": {'values': [512*128, 256*128, 128*128]},
            "NORM_TYPE": {'values': ['layer_norm', 'none']},
            "NORM_QFN": {'values': ['layer_norm', 'none']},
            #"TOTAL_TIMESTEPS": {'values': [100_000_000]},
        },
        'overrides': ['alg=pqn', # 'user=wilka'
                      ],
        'group': 'pqn-3',
    }
  elif search == 'alpha':
    sweep_config = {
        'metric': {
            'name': 'evaluator_performance/0.0 avg_episode_return',
            'goal': 'maximize',
        },
        'parameters': {
            "ENV_NAME": {'values': ['Catch-bsuite']},
        },
        'overrides': ['alg=alphazero',
                      'user=wilka'],
        'group': 'alpha-1',
    }
  elif search == 'dyna':
    sweep_config = {
       'metric': {
            'name': 'evaluator_performance/0.0 avg_episode_return',
            'goal': 'maximize',
        },
        'parameters': {
            # "ENV_NAME": {'values': ['Catch-bsuite']},
            "AGENT_HIDDEN_DIM": {'values': [256]},
            "AGENT_USE_BIAS": {'values': [True]}
        },
        'overrides': ['alg=dyna'],
        'group': 'qlearning-5',
    }
  else:
    raise NotImplementedError(search)

  return sweep_config


@hydra.main(
    version_base=None,
    config_path='configs',
    config_name="config")
def main(config: DictConfig):
  #current_file_path = os.path.abspath(__file__)
  #current_directory = os.path.dirname(current_file_path)
  launcher.run(
      config,
      trainer_filename='__file__',
      config_path='projects/example/configs',
      run_fn=run_single,
      sweep_fn=sweep,
      folder=os.environ.get(
          'RL_RESULTS_DIR', '/tmp/rl_results_dir')
  )


if __name__ == '__main__':
  main()
