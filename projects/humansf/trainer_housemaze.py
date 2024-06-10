"""

TESTING:
JAX_TRACEBACK_FILTERING=off python -m ipdb -c continue projects/humansf/trainer_housemaze.py \
  --debug=True \
  --wandb=True \
  --search=ql

JAX_DISABLE_JIT=1 JAX_TRACEBACK_FILTERING=off python -m ipdb -c continue projects/humansf/trainer_housemaze.py \
  --debug=True \
  --wandb=False \
  --search=alpha

TESTING SLURM LAUNCH:
python projects/humansf/trainer_housemaze.py \
  --parallel=sbatch \
  --debug_parallel=True \
  --search=alpha

RUNNING ON SLURM:
python projects/humansf/trainer_housemaze.py \
  --parallel=sbatch \
  --time '0-02:30:00' \
  --search=alpha
"""
from typing import Any, Callable, Dict, Union, Optional

from absl import flags
from absl import app

import os
import jax
import functools
import jax.numpy as jnp
import jax.tree_util as jtu

from ray import tune

from safetensors.flax import save_file
from flax.traverse_util import flatten_dict

import numpy as np

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import library.flags

from library import parallel
from library import utils
from library import loggers

from projects.humansf import alphazero
from projects.humansf import qlearning
from projects.humansf import offtask_dyna
from projects.humansf import networks
from projects.humansf import observers as humansf_observers
from projects.humansf.housemaze import levels
from projects.humansf.housemaze import renderer
from projects.humansf import housemaze_env as maze

from projects.humansf.housemaze import utils as housemaze_utils
from agents import value_based_basics as vbb

FLAGS = flags.FLAGS


def make_logger(
        config: dict,
        env: maze.HouseMaze,
        env_params: maze.EnvParams,
        action_names: dict,
        render_fn: Callable = None,
        extract_task_info: Callable = None,
        get_task_name: Callable = None,
        learner_log_extra: Optional[Callable[[Any], Any]] = None
):
    return loggers.Logger(
        gradient_logger=loggers.default_gradient_logger,
        learner_logger=loggers.default_learner_logger,
        experience_logger=functools.partial(
            humansf_observers.experience_logger,
            action_names=action_names,
            extract_task_info=extract_task_info,
            get_task_name=get_task_name,
            render_fn=render_fn,
            max_len=config['MAX_EPISODE_LOG_LEN'],
        ),
        learner_log_extra=learner_log_extra,
    )


def load_env_params(
      num_groups: int,
      max_objects: int = 3,
      file: str = 'list_of_groups.npy',
    ):
    # load groups
    if os.path.exists(file):
      list_of_groups = np.load(file)
    else:
      raise RuntimeError(f"Missing file specifying groups for maze: {file}")

    group_set = list_of_groups[0]
    assert num_groups <= 3
    group_set = group_set[:num_groups]

    # load levels
    pretrain_level = levels.two_objects
    train_level = levels.three_pairs_maze1

    ##################
    # create reset parameters
    ##################
    make_int_array = lambda x: jnp.asarray(x, dtype=jnp.int32)
    def make_reset_params(
        map_init,
        train_objects,
        test_objects,
        **kwargs):

      train_objects_ = np.ones(max_objects)*-1
      train_objects_[:len(train_objects)] = train_objects
      test_objects_ = np.ones(max_objects)*-1
      test_objects_[:len(test_objects)] = test_objects
      map_init = map_init.replace(
          grid=make_int_array(map_init.grid),
          agent_pos=make_int_array(map_init.agent_pos),
          agent_dir=make_int_array(map_init.agent_dir),
      )
      return maze.ResetParams(
          map_init=map_init,
          train_objects=make_int_array(train_objects_),
          test_objects=make_int_array(test_objects_),
          **kwargs,
      )
       
    list_of_reset_params = []
    num_starting_locs = 4
    max_starting_locs = 10
    # -------------
    # pretraining levels
    # -------------
    for group in group_set:
      list_of_reset_params.append(
          make_reset_params(
              map_init=maze.MapInit(*housemaze_utils.from_str(
                  pretrain_level, char_to_key=dict(A=group[0], B=group[1]))),
              train_objects=group[:1],
              test_objects=group[1:],
              starting_locs=make_int_array(
                  np.ones((len(group_set), max_starting_locs, 2))*-1)
          )
      )

    # -------------
    # MAIN training level
    # -------------
    train_objects = group_set[:, 0]
    test_objects = group_set[:, 1]
    map_init = maze.MapInit(*housemaze_utils.from_str(
        train_level,
        char_to_key=dict(
            A=group_set[0, 0],
            B=group_set[0, 1],
            C=group_set[1, 0],
            D=group_set[1, 1],
            E=group_set[2, 0],
            F=group_set[2, 1],
        )))

    all_starting_locs = np.ones((len(group_set), max_starting_locs, 2))*-1
    for idx, goal in enumerate(train_objects):
        path = housemaze_utils.find_optimal_path(
            map_init.grid, map_init.agent_pos, np.array([goal]))
        width = len(path)//num_starting_locs
        starting_locs = np.array([path[i] for i in range(0, len(path), width)])
        all_starting_locs[idx, :len(starting_locs)] = starting_locs

    list_of_reset_params.append(
        make_reset_params(
            map_init=map_init,
            train_objects=train_objects,
            test_objects=test_objects,
            starting_locs=make_int_array(all_starting_locs),
            curriculum=jnp.array(True),
        )
    )

    return group_set, maze.EnvParams(
        reset_params=jtu.tree_map(
            lambda *v: jnp.stack(v), *list_of_reset_params),
    )

def run_single(
        config: dict,
        save_path: str = None):

    ###################
    # load data
    ###################
    num_groups = config['env']['ENV_KWARGS'].pop('NUM_GROUPS', 3)
    group_set, env_params = load_env_params(
        num_groups=num_groups,
       file='projects/humansf/housemaze_list_of_groups.npy',
       )
    test_env_params = env_params.replace(training=False)

    image_dict = housemaze_utils.load_image_dict(
        'projects/humansf/housemaze/image_data.pkl')
    # Reshape the images to separate the blocks
    images = image_dict['images']
    reshaped = images.reshape(len(images), 8, 4, 8, 4, 3)

    # Take the mean across the block dimensions
    image_dict['images'] = reshaped.mean(axis=(2, 4)).astype(np.uint8)

    ###################
    # load env
    ###################
    task_objects = group_set.reshape(-1)
    task_runner = maze.TaskRunner(
        task_objects=task_objects)
    keys = image_dict['keys']
    env = maze.HouseMaze(
        task_runner=task_runner,
        num_categories=len(keys),
    )
    env = housemaze_utils.AutoResetWrapper(env)


    ###################
    ## custom observer
    ###################
    action_names = {
        action.value: action.name for action in env.action_enum()}


    def housemaze_render_fn(state: maze.EnvState):
      return renderer.create_image_from_grid(
          state.grid,
          state.agent_pos,
          state.agent_dir,
          image_dict)

    def extract_task_info(timestep: maze.TimeStep):
      state = timestep.state
      return {
          'map_idx': state.map_idx,
          'is_train_task': state.is_train_task,
          'category': state.task_object,
       }

    def task_from_variables(variables):
      map_idx = variables['map_idx']
      category = keys[variables['category']]
      is_train_task = variables['is_train_task']
      setting = '0.pretrain' if map_idx == 0 else '1.main'
      label = '0.train' if is_train_task else '1.test'

      return f'{setting} - {label} - {category}'

    observer_class = functools.partial(
      humansf_observers.TaskObserver,
      extract_task_info=extract_task_info,
      action_names=action_names,
    )

    ##################
    # algorithms
    ##################
    alg_name = config['alg']
    if alg_name == 'qlearning':
      make_train = functools.partial(
          vbb.make_train,
          make_agent=functools.partial(
             qlearning.make_agent,
             ObsEncoderCls=networks.HouzemazeObsEncoder,
             ),
          make_optimizer=qlearning.make_optimizer,
          make_loss_fn_class=qlearning.make_loss_fn_class,
          make_actor=qlearning.make_actor,
          make_logger=functools.partial(
            make_logger,
            render_fn=housemaze_render_fn,
            extract_task_info=extract_task_info,
            get_task_name=task_from_variables,
            action_names=action_names,
            learner_log_extra=functools.partial(
              qlearning.learner_log_extra,
              config=config,
              action_names=action_names,
              extract_task_info=extract_task_info,
              get_task_name=task_from_variables,
              render_fn=housemaze_render_fn,
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

      num_train_simulations = config.get('NUM_SIMULATIONS', 4)
      mcts_policy = functools.partial(
          mctx.gumbel_muzero_policy,
          max_depth=config.get('MAX_SIM_DEPTH', None),
          num_simulations=num_train_simulations,
          gumbel_scale=config.get('GUMBEL_SCALE', 1.0))
      eval_mcts_policy = functools.partial(
          mctx.gumbel_muzero_policy,
          max_depth=config.get('MAX_SIM_DEPTH', None),
          num_simulations=config.get(
            'NUM_EVAL_SIMULATIONS', num_train_simulations),
          gumbel_scale=config.get('GUMBEL_SCALE', 1.0))

      make_train = functools.partial(
          vbb.make_train,
          make_agent=functools.partial(
              alphazero.make_agent,
              ObsEncoderCls=networks.HouzemazeObsEncoder,
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
            render_fn=housemaze_render_fn,
            extract_task_info=extract_task_info,
            get_task_name=task_from_variables,
            action_names=action_names,
            ),
      )
    elif alg_name == 'dynaq':
      import distrax
      temp_dist = distrax.Gamma(
        concentration=config.get("TEMP_CONCENTRATION", 1.),
        rate=config.get("TEMP_RATE", 1.))

      def make_init_offtask_timestep(x: maze.TimeStep, offtask_w: jax.Array):
          task_object = (task_objects*offtask_w).sum(-1)
          task_object = task_object.astype(jnp.int32)
          new_state = x.state.replace(
              step_num=jnp.zeros_like(x.state.step_num),
              task_w=offtask_w,
              task_object=task_object,  # only used for logging
          )

          return x.replace(
              state=new_state,
              observation=jax.vmap(jax.vmap(env.make_observation))(
                  new_state,
                  x.observation.prev_action.argmax(-1),
              ),
              # reset reward, discount, step type
              reward=jnp.zeros_like(x.reward),
              discount=jnp.ones_like(x.discount),
              step_type=jnp.ones_like(x.step_type),
          )


      make_train = functools.partial(
          vbb.make_train,
          make_agent=functools.partial(
            offtask_dyna.make_agent,
            ObsEncoderCls=networks.HouzemazeObsEncoder,
            model_env_params=test_env_params
            ),
          make_optimizer=offtask_dyna.make_optimizer,
          make_loss_fn_class=functools.partial(
            offtask_dyna.make_loss_fn_class,
            temp_dist=temp_dist,
            make_init_offtask_timestep=make_init_offtask_timestep,
            ),
          make_actor=offtask_dyna.make_actor,
          make_logger=functools.partial(
            make_logger,
            render_fn=housemaze_render_fn,
            extract_task_info=extract_task_info,
            get_task_name=task_from_variables,
            action_names=action_names,
            learner_log_extra=functools.partial(
              offtask_dyna.learner_log_extra,
              config=config,
              action_names=action_names,
              extract_task_info=extract_task_info,
              get_task_name=task_from_variables,
              render_fn=housemaze_render_fn,
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
      "config_name": tune.grid_search(['ql_housemaze']),
    }
    space = [
        {
            "group": tune.grid_search(['qlearning-1']),
            "alg": tune.grid_search(['qlearning']),
            "TOTAL_TIMESTEPS": tune.grid_search([30e6]),
            **shared,
        },
      ]
  elif search == 'alpha':
    shared = {
      "config_name": tune.grid_search(['alpha_housemaze']),
    }
    space = [
        {
            "group": tune.grid_search(['alpha-1']),
            "alg": tune.grid_search(['alphazero']),
            "TOTAL_TIMESTEPS": tune.grid_search([5e6]),
            **shared,
        },
      ]
  elif search == 'dynaq':
    shared = {
      "config_name": tune.grid_search(['dyna_housemaze']),
    }
    space = [
        {
            "group": tune.grid_search(['dynaq-1']),
            "alg": tune.grid_search(['dynaq']),
            "TOTAL_TIMESTEPS": tune.grid_search([10e6]),
            "DYNA_COEFF": tune.grid_search([0.1, .01]),
            # "NUM_SIMULATIONS": tune.grid_search([2]),
            # "SIMULATION_LENGTH": tune.grid_search([5, 15]),
            **shared,
            "TEMP_CONCENTRATION": tune.grid_search([.25, .5, 1.]),
            "TEMP_RATE": tune.grid_search([.25, .5]),
        },
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