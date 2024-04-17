"""

TESTING:
JAX_TRACEBACK_FILTERING=off python -m ipdb -c continue projects/humansf/trainer_v1.py \
  --debug=True \
  --wandb=False \
  --search=default

JAX_DISABLE_JIT=1 JAX_TRACEBACK_FILTERING=off python -m ipdb -c continue projects/humansf/trainer_v1.py \
  --debug=True \
  --wandb=False \
  --search=default

TESTING SLURM LAUNCH:
python projects/humansf/trainer_v1.py \
  --parallel=sbatch \
  --debug_parallel=True \
  --search=default

RUNNING ON SLURM:
python projects/humansf/trainer_v1.py \
  --parallel=sbatch \
  --time '0-00:15:00' \
  --search=default
"""
from typing import Dict, Union

from absl import flags
from absl import app

import os
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
from library.wrappers import TimestepWrapper


os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import library.flags
from library import parallel
from projects.humansf import qlearning
from projects.humansf import keyroom
from projects.humansf import keyroom_symbolic
from projects.humansf.minigrid_common import AutoResetWrapper
from projects.humansf import observers
from singleagent import value_based_basics as vbb

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

    # Open the file and load the JSON data
    maze_path = os.path.join('projects/humansf', "maze_pairs.json")
    with open(maze_path, "r") as file:
      maze_config = json.load(file)[0]

    num_rooms = config['env']['ENV_KWARGS'].pop('NUM_ROOMS', 4)
    symbolic = config['env']['ENV_KWARGS'].pop('symbolic', False)

    maze_config = keyroom.shorten_maze_config(
       maze_config, num_rooms)

    if symbolic:
      env = keyroom_symbolic.KeyRoomSymbolic()
      action_names = keyroom_symbolic.get_action_names(maze_config)
    else:
      env = keyroom.KeyRoom()
      action_names = {
          0: 'forward',
          1: 'right',
          2: 'left',
          3: 'pickup',
          4: 'put_down',
          5: 'toggle'}


    env_params = env.default_params(
      maze_config=maze_config, **config['env']['ENV_KWARGS'])
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

    def get_task_name(task_info: dict):
      setting = '0.single' if task_info['room_setting'] == 0 else '1.multi'
      room_idx = task_info['goal_room_idx']
      object_idx = task_info['task_object_idx']
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
            qlearning.make_logger,
            maze_config=maze_config,
            get_task_name=get_task_name,
            action_names=action_names,
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

    #---------------
    # clean up wandb dir
    #---------------
    wandb_dir = wandb_init.get("dir", './wandb')
    if os.path.exists(wandb_dir):
      import shutil
      shutil.rmtree(wandb_dir)

def sweep(search: str = ''):
  search = search or 'default'
  if search == 'default':
    shared = {
      "config_name": tune.grid_search(['ql_keyroom']),
      'env.NUM_ROOMS': tune.grid_search([4, 3, 2, 1]),
    }
    space = [
        #{
        #    "group": tune.grid_search(['qlearning-43-larger?']),
        #    "alg": tune.grid_search(['qlearning']),
        #    'env.symbolic': tune.grid_search([True, False]),
        #    "SAMPLE_LENGTH": tune.grid_search([40]),
        #    "BUFFER_BATCH_SIZE": tune.grid_search([32, 128, 256]),
        #    "TRAINING_INTERVAL": tune.grid_search([1, 10]),
        #    "NUM_ENVS": tune.grid_search([32, 64]),
        #    **shared,
        #},
        {
            "group": tune.grid_search(['qlearning-61-symb']),
            "alg": tune.grid_search(['qlearning']),
            "GAMMA": tune.grid_search([.6]),
            'env.symbolic': tune.grid_search([True]),
            **shared,
        },
        #{
        #    "group": tune.grid_search(['qlearning-55-reg']),
        #    "alg": tune.grid_search(['qlearning']),
        #    'env.symbolic': tune.grid_search([False]),
        #    **shared,
        #},
        #{
        #    "group": tune.grid_search(['qlearning-47-reg-fix']),
        #    "alg": tune.grid_search(['qlearning']),
        #    'env.symbolic': tune.grid_search([False]),
        #    "ENCODER_INIT": tune.grid_search(['word_init', 'word_init2', 'truncated']),
        #    #"BUFFER_BATCH_SIZE": tune.grid_search([32, 128, 256]),
        #    #"TRAINING_INTERVAL": tune.grid_search([1, 10]),
        #    #"NUM_ENVS": tune.grid_search([32, 64]),
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