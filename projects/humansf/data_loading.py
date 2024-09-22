from functools import partial
import polars as pl
import json
import copy
import os.path
from collections import defaultdict
from typing import Callable, NamedTuple, List
from flax import struct
from datetime import datetime
import jax
import jax.numpy as jnp
import numpy as np
from safetensors.flax import load_file
from flax.traverse_util import unflatten_dict
import pickle
import multiprocessing
import jax.tree_util as jtu

from housemaze import utils
from housemaze.human_dyna import multitask_env
from housemaze.human_dyna import mazes
from nicewebrl import nicejax

from jaxneurorl.agents import value_based_basics as vbb

class EpisodeData(NamedTuple):
    actions: jax.Array
    timesteps: multitask_env.TimeStep
    positions: jax.Array = None
    reaction_times: jax.Array = None

def is_in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' in get_ipython().config:
            return True
        else:
            return False
    except ImportError:
        return False
    except AttributeError:
        return False

if is_in_notebook():
    from tqdm.notebook import tqdm
    try:
        import ipywidgets
    except:
        pass
else:
    from tqdm import tqdm

############
# deep learning models
############

@struct.dataclass
class Algorithm:

  config: dict
  train_state: Callable
  actor: Callable
  network: Callable
  reset_fn: Callable
  eval_fn: Callable
  path: str
  name: str

  def seed(self):
     return self.path.split('/')[-1]


def load_params_config(path: str, file: str, config: bool = True):
    filename = f'{path}/{file}.safetensors'
    flattened_dict = load_file(filename)
    params = unflatten_dict(flattened_dict, sep=',')

    if config:
        with open(f'{path}/{file}.config', 'rb') as f:
            config = pickle.load(f)
    return params, config

def swap_task(x: multitask_env.TimeStep, w: jax.Array):
    new_state = x.state.replace(
        step_num=jnp.zeros_like(x.state.step_num),
        task_w=w,
    )

    return x.replace(
        state=new_state,
    )

def load_algorithm(
      path,
      name,
      example_env_params,
      env,
      make_fns,
      nenvs: int=25,
      ):
  agent_params, config = load_params_config(path, name)

  config['NUM_ENVS'] = nenvs

  def vmap_reset(rng, env_params):
    return jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, config["NUM_ENVS"]), env_params)

  def vmap_step(rng, env_state, action, env_params):
      return jax.vmap(
          env.step, in_axes=(0, 0, 0, None))(
          jax.random.split(rng, config["NUM_ENVS"]), env_state, action, env_params)

  rng = jax.random.PRNGKey(config["SEED"])
  rng, rng_ = jax.random.split(rng)
  example_timestep = vmap_reset(rng_, example_env_params)

  fns = make_fns(config=config)

  network, _, reset_fn = fns.make_agent(
      config=config,
      env=env,
      env_params=example_env_params,
      example_timestep=example_timestep,
      rng=rng_)

  actor = fns.make_actor(
              config=config,
              agent=network,
              rng=rng_)

  train_state = vbb.CustomTrainState.create(
              apply_fn=network.apply,
              params=agent_params,
              target_network_params=agent_params,
              tx=fns.make_optimizer(config),  # unnecessary
          )

  def collect_trajectory(
      init_timestep,
      task_w,
      rng,
      max_steps: int = 50,
      ):
      rng, rng_ = jax.random.split(rng)

      timestep = jax.vmap(swap_task, (0, None))(init_timestep, task_w)
      agent_state = reset_fn(train_state.params, timestep, rng_)

      runner_state = vbb.RunnerState(
          train_state=train_state,
          timestep=timestep,
          agent_state=agent_state,
          rng=rng)

      _, traj_batch = vbb.collect_trajectory(
                  runner_state=runner_state,
                  num_steps=max_steps,
                  actor_step_fn=actor.eval_step,
                  env_step_fn=vmap_step,
                  env_params=example_env_params)

      return EpisodeData(
         timesteps=traj_batch.timestep,
         actions=traj_batch.action)

  def collect_trajectories(
        rng,
        env_params,
        task,
        n=1):

      def scan_body(rng, _):
          rng, rng_ = jax.random.split(rng)
          init_timestep = vmap_reset(rng=rng_, env_params=env_params)
          rng, rng_ = jax.random.split(rng)
          new_trajs = collect_trajectory(init_timestep, task, rng_)
          return rng, new_trajs

      # [n, num_timesteps, num_traj, data]
      rng, all_trajs = jax.lax.scan(scan_body, rng, None, length=n)
      def fix_shape(x):
        # [n, num_traj, num_timesteps, data]
        x = jnp.swapaxes(x, 1, 2)

        # [n*num_traj, num_timesteps, data]
        x = x.reshape(-1, *x.shape[2:])
        return x

      # [n*num_traj, num_timesteps, data]
      all_trajs = jax.tree_map(fix_shape, all_trajs)
      return all_trajs

  return Algorithm(
      config=config,
      network=network,
      reset_fn=reset_fn,
      actor=actor,
      train_state=train_state,
      eval_fn=jax.jit(collect_trajectories),
      path=path,
      name=name,
  )

############
# Human data
############
image_data = utils.load_image_dict()
image_keys = image_data['keys']

groups = [
    # room 1
    [image_keys.index('orange'), image_keys.index('potato')],
    # room 2
    [image_keys.index('knife'), image_keys.index('spoon')],
    # room 3
    [image_keys.index('tomato'), image_keys.index('lettuce')],
]
groups = np.array(groups, dtype=np.int32)
task_objects = groups.reshape(-1)

def get_timestep(datum, example_timestep):
    timestep = nicejax.deserialize_bytes(
        cls=multitask_env.TimeStep, encoded_data=datum['data'])

    # `deserialize_bytes` infers the types so it might be slightly wrong. you can enforce the correct types by matching them to example data.
    timestep = nicejax.match_types(
        example=example_timestep, data=timestep)

    return timestep

def time_diff(t1, t2) -> float:
    # Convert string timestamps to datetime objects
    t1 = datetime.strptime(t1, '%Y-%m-%dT%H:%M:%S.%fZ')
    t2 = datetime.strptime(t2, '%Y-%m-%dT%H:%M:%S.%fZ')

    # Calculate the time difference
    time_difference = t2 - t1

    # Convert the time difference to milliseconds
    return time_difference.total_seconds() * 1000

def compute_reaction_time(datum) -> float:
    # Calculate the time difference
    return time_diff(datum['image_seen_time'], datum['action_taken_time'])

def get_task_object(timesteps: multitask_env.TimeStep):
    return timesteps.state.task_object[0]

def get_task_room(timesteps: multitask_env.TimeStep):
    task_object = get_task_object(timesteps)
    # Find the room (row) that contains the task object
    task_room = next((i for i, row in enumerate(groups) if task_object in row), None)
    return task_room

def dict_to_string(data):
    # Convert each key-value pair to "key=value" format
    pairs = [f"{key}={value}" for key, value in data.items()]

    # Join all pairs with ", " separator
    return ", ".join(pairs)

def get_block_stage_description(datum):
    ####################
    # block information
    ####################
    block_metadata = datum['metadata']['block_metadata']
    # e.g. manipulation = 4
    block_manipulation = block_metadata.get('manipulation', -1)
    # e.g. desc = 'off-task object regular'
    block_desc = block_metadata.get('desc', 'unknown')

    ####################
    # stage information
    ####################
    stage_desc = datum['metadata'].get('desc')

    return dict(
        stage=stage_desc,
        block=block_desc,
        manipulation=block_manipulation,
        episode_idx=datum['metadata']['episode_idx'],
        eval=datum['metadata']['eval'],
    )

def separate_data_by_block_stage(data: List[dict]):
    """This function will group episodes by the values from get_block_stage_description

    The input i
    So for example, each episode with {'stage': "'not obvious' shortcut",
     'block': 'shortcut',
     'manipulation': 1,
     'episode_idx': 1,
     'eval': True}
     with go into its own list.
    """
    grouped_data = defaultdict(list)
    episode_idx = -1
    keys = set()
    infos = dict()
    # first group all of the data based on which (stage, block) its in
    for datum in data:
        info = get_block_stage_description(datum)
        key = dict_to_string(info)
        if not key in keys:
            episode_idx += 1
            keys.add(key)
        info['user_episode_idx'] = episode_idx
        
        updated_key = dict_to_string(info)
        grouped_data[updated_key].append(datum)
        infos[updated_key] = info
    return grouped_data, infos


def make_episode_data(data: List[dict], example_timestep: multitask_env.TimeStep):
    """This groups all of the data by block/stage information and prepares 
        (1) a list of EpisodeData objects per block/stage
        (2) a dataframe which summarizes all episode information.

    The dataframe can be used to get indices into the list of EpisodeData for further computation.
    """
    def filter_fn(datum):
        if 'metadata' not in datum: return True
        desc = datum['metadata']['block_metadata']['desc']
        if 'practice' in desc: return True
    nbefore = len(data)
    data = [datum for datum in data if not filter_fn(datum)]
    print(f"Filtered {nbefore-len(data)} data points")
    gds, gd_infos = separate_data_by_block_stage(data)

    episode_data = [None]*len(gds.keys())
    episode_info = [None]*len(gds.keys())
    for key in tqdm(gds.keys(), desc="Processing episodes"):
        red = raw_episode_data = gds[key]
        # get actions

        actions = jnp.asarray([datum['action_idx'] for datum in red])

        # collect timesteps
        timesteps = [get_timestep(datum, example_timestep) for datum in red]
        
        # combine them into trajectory
        timesteps = jtu.tree_map(
                lambda *v: jnp.stack(v), *timesteps)

        positions = timesteps.state.agent_pos

        reaction_times = [compute_reaction_time(datum) for datum in red]
        reaction_times = jnp.asarray(reaction_times)

        episode_idx = gd_infos[key]['user_episode_idx']
        episode_data[episode_idx] = EpisodeData(
            actions=actions,
            positions=positions,
            reaction_times=reaction_times,
            timesteps=timesteps,

        )

        ######################
        # THIS IS WHERE YOU'LL WANT TO INSERT OTHER EPISODE LEVEL INFO TO TRACK IN DATAFRAME!!!
        ######################
        info = copy.deepcopy(gd_infos[key])
        info.update(
            task=get_task_object(timesteps),
            room=get_task_room(timesteps),
        )
        datum0 = red[0]
        # add in user information to dataframe
        info.update(datum0['user_data'])

        episode_info[episode_idx] = info

    episode_info = pl.DataFrame(episode_info)
    return episode_info, episode_data

def make_all_episode_data(files, example_timestep, base_path, filter_fn = None, overwrite: bool = False):

    all_episode_data = []
    episode_df_list = []
    for file in tqdm(files, desc="Processing files"):
        user_filename = file.split("/")
        user_filename = file.split("/")[1].split(".json")[0]
        timesteps_filename = f"{base_path}/{user_filename}_timesteps.pickle"
        df_filename = f"{base_path}/{user_filename}_df.csv"
        if (os.path.exists(timesteps_filename) and os.path.exists(df_filename) and not overwrite):
            episode_df = pl.read_csv(df_filename)
            with open(timesteps_filename, 'rb') as f:
                episode_data = pickle.load(f)
        else:
            with open(file, 'r') as f:
                data = json.load(f)

            if filter_fn is not None:
                if filter_fn(data): continue
            episode_df, episode_data = make_episode_data(data, example_timestep)
            episode_df.write_csv(df_filename)
            with open(timesteps_filename, 'wb') as f:
                pickle.dump(episode_data, f)
        all_episode_data += episode_data
        episode_df_list.append(episode_df)

    episode_df = pl.concat(episode_df_list).with_row_count(name="index").with_columns(
    pl.col("index").add(1).alias("index"))

    return episode_df, all_episode_data
