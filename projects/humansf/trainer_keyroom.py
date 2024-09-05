"""

TESTING:
JAX_TRACEBACK_FILTERING=off python -m ipdb -c continue projects/humansf/trainer_v1.py \
  --debug=True \
  --wandb=False \
  --search=alpha

JAX_DISABLE_JIT=1 JAX_TRACEBACK_FILTERING=off python -m ipdb -c continue projects/humansf/trainer_v1.py \
  --debug=True \
  --wandb=False \
  --search=alpha

TESTING SLURM LAUNCH:
python projects/humansf/trainer_v1.py \
  --parallel=sbatch \
  --debug_parallel=True \
  --search=alpha

RUNNING ON SLURM:
python projects/humansf/trainer_v1.py \
  --parallel=sbatch \
  --time '0-08:00:00' \
  --search=alpha
"""
from typing import Any, Callable, Dict, Union, Optional

from absl import flags
from absl import app

import os
from gymnax.environments import environment
import jax
import json
import functools

from ray import tune

import wandb
from safetensors.flax import save_file
from flax.traverse_util import flatten_dict

import hydra
import gymnax
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper
from jaxneurorl.wrappers import TimestepWrapper
from projects.humansf import logger, observers as humansf_observers


os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import jaxneurorl.flags

from jaxneurorl import loggers, parallel
from jaxneurorl import utils

from projects.humansf import keyroom
from projects.humansf import keyroom_symbolic
from projects.humansf.minigrid_common import AutoResetWrapper
from projects.humansf import observers
from projects.humansf import alphazero
from projects.humansf import qlearning
from projects.humansf import offtask_dyna

from jaxneurorl.agents import value_based_basics as vbb

FLAGS = flags.FLAGS


def make_logger(
        config: dict,
        env: environment.Environment,
        env_params: environment.EnvParams,
        maze_config: dict,
        action_names: dict,
        get_task_name: Callable = None,
        learner_log_extra: Optional[Callable[[Any], Any]] = None
):
    return loggers.Logger(
        gradient_logger=loggers.default_gradient_logger,
        learner_logger=loggers.default_learner_logger,
        experience_logger=functools.partial(
            humansf_observers.experience_logger,
            action_names=action_names,
            get_task_name=get_task_name,
            max_len=config['MAX_EPISODE_LOG_LEN'],
        ),
        learner_log_extra=learner_log_extra,
    )

def run_single(
        config: dict,
        save_path: str = None):

    # Open the file and load the JSON data
    maze_path = os.path.join('projects/humansf', "maze_pairs.json")
    with open(maze_path, "r") as file:
      maze_config = json.load(file)[0]

    num_rooms = config['rlenv']['ENV_KWARGS'].pop('NUM_ROOMS', 3)
    num_tiles = config['rlenv']['ENV_KWARGS'].pop('NUM_TILES', 16)
    train_end_pair = config['rlenv']['ENV_KWARGS'].pop('TRAIN_END_PAIR', True)
    test_end_on = config['rlenv']['ENV_KWARGS'].pop('TEST_END_ON', 'any_pair')
    symbolic = config['rlenv']['ENV_KWARGS'].pop('symbolic', False)

    maze_config = keyroom.shorten_maze_config(
       maze_config, num_rooms)

    default_params_kwargs = dict(
      maze_config=maze_config,
      height=num_tiles,
      width=num_tiles,
      **config['rlenv']['ENV_KWARGS'])

    if symbolic:
      env = keyroom_symbolic.KeyRoomSymbolic()
      env_params = env.default_params(**default_params_kwargs)
      action_names = keyroom_symbolic.get_action_names(env_params)
    else:
      env = keyroom.KeyRoom(
        train_episode_ends_on_pair_pickup=train_end_pair,
        test_episode_ends_on=test_end_on)

      env_params = env.default_params(**default_params_kwargs)
      action_names = {
          0: 'forward',
          1: 'right',
          2: 'left',
          3: 'pickup',
          4: 'put_down',
          5: 'toggle'}

    test_env_params = env_params.replace(training=False)

    # auto-reset wrapper
    env = AutoResetWrapper(env)

    ##################
    # custom observer
    ##################
    def extract_task_info(timestep: keyroom.TimeStep):
      state: keyroom.EnvState = timestep.state
      return {
          'room_setting': state.room_setting,
          'goal_room_idx': state.goal_room_idx,
          'task_object_idx': state.task_object_idx,
       }

    def get_task_name(room_setting, goal_room_idx, task_object_idx):
      setting = '0.single' if room_setting == 0 else '1.multi'
      room_idx = goal_room_idx
      object_idx = task_object_idx
      label = '1.test' if object_idx > 0 else '0.train'

      category, color = maze_config['pairs'][room_idx][object_idx]
      return f'{setting} - {label} - {color} {category}'

    observer_class = functools.partial(
      observers.TaskObserver,
      extract_task_info=extract_task_info,
      get_task_name=get_task_name,
      action_names=action_names,
    )

    ##################
    # algorithms
    ##################
    alg_name = config['alg']
    if alg_name == 'qlearning':
      make_train = functools.partial(
          vbb.make_train,
          make_agent=qlearning.make_agent,
          make_optimizer=qlearning.make_optimizer,
          make_loss_fn_class=qlearning.make_loss_fn_class,
          make_actor=qlearning.make_actor,
          make_logger=functools.partial(
            make_logger,
            maze_config=maze_config,
            get_task_name=get_task_name,
            action_names=action_names,
            learner_log_extra=functools.partial(
              qlearning.learner_log_extra,
              config=config,
              action_names=action_names,
              maze_config=maze_config,
              )
            ),
      )
    elif alg_name == 'alphazero':
      import mctx
      max_value = config.get('MAX_VALUE', 10)
      num_bins = config['NUM_BINS']

      discretizer = utils.Discretizer(
          max_value=max_value,
          num_bins=num_bins,
          min_value=-max_value)

      search_type = config.get('SEARCH_TYPE', 'gumbel')
      num_train_simulations = config.get('NUM_SIMULATIONS', 4)
      if search_type == 'gumbel':
        train_gumbel_scale = config.get('GUMBEL_SCALE', 1.0)
        mcts_policy = functools.partial(
            mctx.gumbel_muzero_policy,
            max_depth=config.get('MAX_SIM_DEPTH', None),
            num_simulations=num_train_simulations,
            gumbel_scale=train_gumbel_scale)
        eval_mcts_policy = functools.partial(
            mctx.gumbel_muzero_policy,
            max_depth=config.get('MAX_SIM_DEPTH', None),
            num_simulations=config.get(
              'NUM_EVAL_SIMULATIONS', num_train_simulations),
            gumbel_scale=config.get(
              'EVAL_GUMBEL_SCALE', train_gumbel_scale))
      elif search_type == 'vanilla':
        mcts_policy = functools.partial(
            mctx.muzero_policy,
            max_depth=config.get('MAX_SIM_DEPTH', None),
            num_simulations=num_train_simulations)
        eval_mcts_policy = functools.partial(
            mctx.muzero_policy,
            max_depth=config.get('MAX_SIM_DEPTH', None),
            num_simulations=config.get(
              'NUM_EVAL_SIMULATIONS', num_train_simulations),
            dirichlet_fraction=config.get('EVAL_FRACTION_DIRICHLET', .25),
            dirichlet_alpha=config.get('EVAL_ALPHA_DIRICHLET', .3),
            )

      make_train = functools.partial(
          vbb.make_train,
          make_agent=functools.partial(
              alphazero.make_agent,
              test_env_params=test_env_params),
          make_optimizer=alphazero.make_optimizer,
          make_loss_fn_class=functools.partial(
              alphazero.make_loss_fn_class,
              discretizer=discretizer),
          make_actor=functools.partial(
              alphazero.make_actor,
              discretizer=discretizer,
              mcts_policy=mcts_policy,
              eval_mcts_policy=eval_mcts_policy),
          make_logger=functools.partial(
            make_logger,
            maze_config=maze_config,
            get_task_name=get_task_name,
            action_names=action_names,
            ),
      )
    elif alg_name == 'dynaq':
      import distrax
      temp_dist = distrax.Gamma(
        concentration=config.get("TEMP_CONCENTRATION", 1.),
        rate=config.get("TEMP_RATE", 1.))
      make_train = functools.partial(
          vbb.make_train,
          make_agent=functools.partial(
            offtask_dyna.make_agent,
            model_env_params=test_env_params
            ),
          make_optimizer=offtask_dyna.make_optimizer,
          make_loss_fn_class=functools.partial(
            offtask_dyna.make_loss_fn_class,
            temp_dist=temp_dist,
            env_params=test_env_params,  # TODO: this is technically cheating? but okay?
            ),
          make_actor=offtask_dyna.make_actor,
          make_logger=functools.partial(
            make_logger,
            maze_config=maze_config,
            get_task_name=get_task_name,
            action_names=action_names,
            learner_log_extra=functools.partial(
              offtask_dyna.learner_log_extra,
              config=config,
              action_names=action_names,
              maze_config=maze_config,
              )
            ),
      )

    else:
      raise NotImplementedError(alg_name)

    train_fn = make_train(
      config=config,
      env=env,
      train_env_params=env_params,
      test_env_params=test_env_params,
      ObserverCls=observer_class,
      )
    train_vjit = jax.jit(jax.vmap(train_fn))

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    outs = jax.block_until_ready(train_vjit(rngs))

    #---------------
    # save model weights
    #---------------
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
  search = search or 'ql'
  if search == 'ql':
    shared = {
      "config_name": tune.grid_search(['ql_keyroom']),
    }
    space = [
        {
            "group": tune.grid_search(['qlearning-82']),
            "alg": tune.grid_search(['qlearning']),
            **shared,
        },
      ]
  elif search == 'alpha-gumbel':
    shared = {
      "config_name": tune.grid_search(['alpha_keyroom']),
    }
    space = [
        {
            "group": tune.grid_search(['alpha-13-gumbel']),
            "alg": tune.grid_search(['alphazero']),
            "EVAL_GUMBEL_SCALE": tune.grid_search([0, 1.0, 10.0, 100.]),
            **shared,
        },
      ]
  elif search == 'alpha-vanilla':
    shared = {
      "config_name": tune.grid_search(['alpha_keyroom']),
    }
    space = [
        {
            "group": tune.grid_search(['alpha-13-vanilla']),
            "alg": tune.grid_search(['alphazero']),
            "EVAL_FRACTION_DIRICHLET": tune.grid_search([.25, .75, 1.]),
            "EVAL_ALPHA_DIRICHLET": tune.grid_search([.3, .1]),
            **shared,
        },
      ]
  elif search == 'symbolic':
    space = [
        {
            "group": tune.grid_search(['symbolic-2']),
            "alg": tune.grid_search(['alphazero']),
            "env.symbolic": tune.grid_search([True]),
            "config_name": tune.grid_search(['alpha_keyroom']),
        },
        {
            "group": tune.grid_search(['symbolic-2']),
            "alg": tune.grid_search(['qlearning']),
            "env.symbolic": tune.grid_search([True]),
            "config_name": tune.grid_search(['ql_keyroom']),
        },
      ]
  elif search == 'dynaq':
    shared = {
      "config_name": tune.grid_search(['dyna_keyroom']),
    }
    space = [
        #{
        #    "group": tune.grid_search(['dyna-temp-2']),
        #    "alg": tune.grid_search(['dynaq']),
        #    "DYNA_COEFF": tune.grid_search([0.1]),
        #    "TEMP_CONCENTRATION": tune.grid_search([.5, 1.]),
        #    "TEMP_RATE": tune.grid_search([.5, 1]),
        #    #"NUM_SIMULATIONS": tune.grid_search([2]),
        #    #"SIMULATION_LENGTH": tune.grid_search([5, 15]),
        #    **shared,
        #},
        {
            "group": tune.grid_search(['dyna-coeff-9-i5']),
            "alg": tune.grid_search(['dynaq']),
            "DYNA_COEFF": tune.grid_search([1., 0.1]),
            # "NUM_SIMULATIONS": tune.grid_search([2]),
            # "SIMULATION_LENGTH": tune.grid_search([5, 15]),
             "TRAINING_INTERVAL": tune.grid_search([5]),
            **shared,
            "TEMP_CONCENTRATION": tune.grid_search([.25]),
            "TEMP_RATE": tune.grid_search([.25, .5]),
        },
        {
            "group": tune.grid_search(['dyna-coeff-9-i1']),
            "alg": tune.grid_search(['dynaq']),
            "DYNA_COEFF": tune.grid_search([1., 0.1]),
            # "NUM_SIMULATIONS": tune.grid_search([2]),
            # "SIMULATION_LENGTH": tune.grid_search([5, 15]),
             "TRAINING_INTERVAL": tune.grid_search([1]),
            **shared,
            "TEMP_CONCENTRATION": tune.grid_search([.25]),
            "TEMP_RATE": tune.grid_search([.25, .5]),
        },
        #{
        #    "group": tune.grid_search(['dyna-grad-4']),
        #    "alg": tune.grid_search(['dynaq']),
        #    "STOP_DYNA_GRAD": tune.grid_search([True, False]),
        #    **shared,
        #},
        #{
        #    "group": tune.grid_search(['dyna-sanity-2']),
        #    "alg": tune.grid_search(['dynaq']),
        #    "DYNA_COEFF": tune.grid_search([0.0]),
        #    **shared,
        #},
      ]
  else:
    raise NotImplementedError(search)

  return space

def main(_):
  parallel.run(
      trainer_filename=__file__,
      config_path='projects/humansf/configs',
      run_fn=run_single,
      sweep_fn=sweep,
      folder=os.environ.get(
          'RL_RESULTS_DIR', '/tmp/rl_results_dir')
  )

if __name__ == '__main__':
  app.run(main)

