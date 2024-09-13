import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os.path
import numpy as np

import jax.tree_util as jtu
import polars as pl
import pickle

from housemaze import renderer
from housemaze.human_dyna import utils
from housemaze.human_dyna import mazes
from housemaze.human_dyna import multitask_env

from projects.humansf import housemaze_experiments
from projects.humansf import data_loading

image_dict = utils.load_image_dict()

num_groups = 2
char2idx, groups, task_objects = mazes.get_group_set(num_groups)

task_runner = multitask_env.TaskRunner(task_objects=task_objects)


############
# trained model
############
def get_params(maze_str: str = None):
  maze_str = maze_str or mazes.maze0
  return mazes.get_maze_reset_params(
      groups=groups,
      char2key=char2idx,
      maze_str=maze_str,
      randomize_agent=False,
      make_env_params=True,
)

def get_algorithm_data(
        algorithm: data_loading.Algorithm,
        exp: str,
        overwrite: bool = False,
      ):
  exp_fn = getattr(housemaze_experiments, exp, None)
  _, _, _, label2name = exp_fn(algorithm.config, analysis_eval=True)
  
  base_path = f"{algorithm.path}/analysis/"
  os.makedirs(base_path, exist_ok=True)
  timesteps_filename = f"{base_path}/{algorithm.name}_timesteps.pickle"
  df_filename = f"{base_path}/{algorithm.name}_df.csv"
  ##############################
  # if already exists, return
  ##############################
  if (os.path.exists(timesteps_filename) and os.path.exists(df_filename) and not overwrite):
    df = pl.read_csv(df_filename)
    with open(timesteps_filename, 'rb') as f:
      all_episodes = pickle.load(f)

    return df, all_episodes

  ##############################
  # create data and return
  ##############################

  rng = jax.random.PRNGKey(42)
  train_task = groups[0, 0]
  test_task = groups[1, 0]

  all_info = []
  all_episodes = []
  for maze_name in label2name.values():
      env_params = get_params(getattr(mazes, maze_name))
      for task in [train_task, test_task]:
          task_vector = task_runner.task_vector(task)
          episodes = algorithm.eval_fn(rng, env_params, task_vector)
          info = dict(
              eval=bool(task==test_task),
              algo=algorithm.name,
              exp=exp,
              room=0,
              task=task,
              maze_name=maze_name,
          )
          #print("-"*50)
          #print("Finished")
          #print(info)
          all_info.append(info)
          all_episodes.append(episodes)
  df = pl.DataFrame(all_info)
  df.write_csv(df_filename)

  with open(timesteps_filename, 'wb') as f:
    pickle.dump(all_episodes, f)
  
  return df, all_episodes

###################
# Search
###################

def concat_pytrees(tree1, tree2, **kwargs):
    return jax.tree_map(lambda x, y: jnp.concatenate((x, y), **kwargs), tree1, tree2)
def add_time(v): return jax.tree_map(lambda x: x[None], v)
def concat_first_rest(first, rest):
    # init: [...]
    # rest: [T, ...]
    # output: [T+1, ...]
    return concat_pytrees(add_time(first), rest)

def actions_from_search(env_params, rng, task, algo, budget):
    map_init = jax.tree_map(lambda x:x[0], env_params.reset_params.map_init)
    grid = np.asarray(map_init.grid)
    agent_pos = tuple(int(o) for o in map_init.agent_pos)
    goal = np.array([task])
    path, _ = algo(grid, agent_pos, goal, key=rng, budget=budget)
    actions = utils.actions_from_path(path)
    return actions

def collect_search_episodes(
  env, env_params, task, algorithm: str, budget=None,
  max_steps: int =50, 
  n: int=100):
  budget = budget or 1e8

  def step_fn(carry, action):
      rng, timestep = carry
      rng, step_rng = jax.random.split(rng)
      next_timestep = env.step(step_rng, timestep, action, env_params)
      return (rng, next_timestep), next_timestep

  def collect_episode(task, actions, rng):
    timestep = env.reset(rng, env_params)
    task_vector = task_runner.task_vector(task)
    timestep = data_loading.swap_task(timestep, task_vector)
    initial_carry = (rng, timestep)
    (rng, timestep), timesteps = jax.lax.scan(step_fn, initial_carry, actions)
    return concat_first_rest(timestep, timesteps)

  #######################
  # first get actions from n different runs
  #######################
  all_episodes = []
  all_actions = []
  rng = jax.random.PRNGKey(42)
  rngs = jax.random.split(rng, n)
  for idx in range(n):
      actions = actions_from_search(
         env_params, rngs[idx], task,
         algo=getattr(utils, algorithm),
         budget=budget)
      leftover = max_steps - len(actions) + 1
      actions = np.concatenate((actions, np.array([0]*leftover)))
      actions = actions.astype(np.int32)
      episode = collect_episode(task, actions[:-1], rngs[idx])
      # might be variable length
      all_episodes.append(episode)
      all_actions.append(actions)

  # [N, T]
  all_actions = np.array(all_actions)
  all_episodes = jtu.tree_map(lambda *v: jnp.stack(v), *all_episodes)
  return data_loading.EpisodeData(
    timesteps=all_episodes,
    actions=all_actions)


def get_search_data(
        algorithm: str,
        env,
        exp: str,
        base_path: str,
        budget: int = None,
        overwrite: bool = False,
        searches: int=100,
      ):

  exp_fn = getattr(housemaze_experiments, exp, None)
  _, _, _, label2name = exp_fn({}, analysis_eval=True)

  os.makedirs(base_path, exist_ok=True)
  timesteps_filename = f"{base_path}/{algorithm}_{budget}_timesteps.pickle"
  df_filename = f"{base_path}/{algorithm}_{budget}_df.csv"

  ##############################
  # if already exists, return
  ##############################
  if (os.path.exists(timesteps_filename) and os.path.exists(df_filename) and not overwrite):
    df = pl.read_csv(df_filename)
    with open(timesteps_filename, 'rb') as f:
      all_episodes = pickle.load(f)

    return df, all_episodes

  ##############################
  # create data and return
  ##############################

  train_task = groups[0, 0]
  test_task = groups[1, 0]

  all_info = []
  all_episodes = []
  for maze_name in label2name.values():
      env_params = get_params(getattr(mazes, maze_name))
      for task in [train_task, test_task]:
          episodes = collect_search_episodes(
             env=env,
             env_params=env_params,
             task=task,
             algorithm=algorithm,
             budget=budget,
             n=searches)
          info = dict(
              eval=bool(task==test_task),
              algo=algorithm,
              exp=exp,
              room=0,
              task=task,
              budget=budget
          )
          #print("-"*50)
          #print("Finished")
          #print(info)
          all_info.append(info)
          all_episodes.append(episodes)

  df = pl.DataFrame(all_info)
  df.write_csv(df_filename)

  with open(timesteps_filename, 'wb') as f:
    pickle.dump(all_episodes, f)
  
  return df, all_episodes


###################
# Visualizations
###################

def get_in_episode(timestep):
  # get mask for within episode
  non_terminal = timestep.discount
  is_last = timestep.last()
  term_cumsum = jnp.cumsum(is_last, -1)
  in_episode = (term_cumsum + non_terminal) < 2
  return in_episode

def housemaze_render_fn(state: multitask_env.EnvState):
    return renderer.create_image_from_grid(
        state.grid,
        state.agent_pos,
        state.agent_dir,
        image_dict)

def render_path(episode_data, from_model=True, ax=None):
    # get actions that are in episode
    timesteps = episode_data.timesteps
    actions = episode_data.actions
    if from_model:
      in_episode = get_in_episode(timesteps)
      actions = actions[in_episode][:-1]
      positions = jax.tree_map(lambda x: x[in_episode][:-1], timesteps.state.agent_pos)
    else:
       positions = timesteps.state.agent_pos[:-1]
    # positions in episode

    state_0 = jax.tree_map(lambda x: x[0], timesteps.state)

    # doesn't matter
    maze_height, maze_width, _ = timesteps.state.grid[0].shape

    if ax is None:
      fig, ax = plt.subplots(1, figsize=(5, 5))
    img = housemaze_render_fn(state_0)
    renderer.place_arrows_on_image(img, positions, actions, maze_height, maze_width, arrow_scale=5, ax=ax)
