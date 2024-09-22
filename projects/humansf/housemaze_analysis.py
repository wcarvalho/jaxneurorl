import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os.path
import numpy as np
from matplotlib.animation import FuncAnimation

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
idx2key = {idx: image_dict['keys'][idx] for char, idx in char2idx.items()}
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
        extra_info = None,
      ):
  extra_info = extra_info or {}
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
  train_tasks = groups[:1, 0]
  test_tasks = groups[:1, 1]
  tasks = jnp.concatenate((train_tasks, test_tasks))

  all_info = []
  all_episodes = []
  for maze_name in label2name.values():
      env_params = get_params(getattr(mazes, maze_name))
      for task in tasks:
          task_vector = task_runner.task_vector(task)
          episodes = algorithm.eval_fn(rng, env_params, task_vector)
          info = dict(
              eval=bool(task in test_tasks),
              algo=algorithm.name,
              exp=exp,
              room=0,
              task=task,
              maze_name=maze_name,
              **extra_info,
          )

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
  max_steps: int = 100, 
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
  all_actions = []
  rng = jax.random.PRNGKey(42)
  rngs = jax.random.split(rng, n)

  # First, get all actions
  for idx in range(n):
      actions = actions_from_search(
          env_params, rngs[idx], task,
          algo=getattr(utils, algorithm),
          budget=budget
      )
      all_actions.append(actions)

  # Find the maximum length among all action sequences
  max_length = max(len(actions) for actions in all_actions)

  # Pad each action sequence to the maximum length
  padded_actions = []
  for actions in all_actions:
      padding = [0] * (max_length - len(actions))
      padded_actions.append(np.concatenate((actions, np.array(padding, dtype=np.int32))))

  # Convert to numpy array
  all_actions = np.array(padded_actions, dtype=np.int32)

  # Now compute all episodes
  all_episodes = []
  for idx in range(n):
      episode = collect_episode(task, all_actions[idx][:-1], rngs[idx])
      all_episodes.append(episode)

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

  train_tasks = groups[:1, 0]
  test_tasks = groups[:1, 1]
  tasks = jnp.concatenate((train_tasks, test_tasks))

  all_info = []
  all_episodes = []
  for maze_name in label2name.values():
      env_params = get_params(getattr(mazes, maze_name))
      for task in tasks:
          episodes = collect_search_episodes(
             env=env,
             env_params=env_params,
             task=task,
             algorithm=algorithm,
             budget=budget,
             n=searches)
          info = dict(
              eval=bool(task in test_tasks),
              algo=algorithm,
              exp=exp,
              room=0,
              task=task,
              budget=budget,
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


def create_reaction_times_video(images, reaction_times, output_file, fps=1):
    # Ensure the directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n = len(images)
    width = 5
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2*width, width))

    def update(frame):
        # Clear previous content
        ax1.clear()
        ax2.clear()

        # Left plot: Image
        if images.size > 0:
            img = images[frame]
            ax1.imshow(img, cmap='viridis')
        else:
            ax1.text(0.5, 0.5, "No image data", ha='center', va='center')
        rt = reaction_times[frame]/1e3
        ax1.set_title(
            f"Step: {frame}, Reaction Time: {rt:.2f} s")
        ax1.axis('off')

        # Right plot: Bar plot of reaction times
        bars = ax2.bar(range(len(reaction_times)),
                       reaction_times, color='lightblue')
        bars[frame].set_color('red')  # Highlight current index
        ax2.set_xlabel('Time Index')
        ax2.set_ylabel('Reaction Time')
        ax2.set_title('Reaction Times')
        ax2.set_ylim(0, max(reaction_times) * 1.1)

        return ax1, ax2

    # Create the animation
    anim = FuncAnimation(fig, update, frames=n, interval=1000/fps, blit=False)
    video = anim.to_html5_video()
    return video


def create_episode_reaction_times_video(
      episode_data,
      output_file='/tmp/housemaze_anlaysis/rt_video.mp4',
      fps=1,
      html: bool = True,
      ):
  images = jax.vmap(housemaze_render_fn)(episode_data.timesteps.state)
  reaction_times = episode_data.reaction_times
  video = create_reaction_times_video(images, reaction_times, output_file, fps)
  if html:
     from IPython.display import HTML, display
     return display(HTML(video))
  return video
###################
# Metrics
###################


def success(e):
    rewards = e.timesteps.reward
    #return rewards
    assert rewards.ndim == 1, 'this is only defined over vector, e.g. 1 episode'
    success = rewards > .5
    return success.any().astype(np.float32)


def rewards(e):
    rewards = e.timesteps.reward
    assert rewards.ndim == 1, 'this is only defined over vector, e.g. 1 episode'
    success = rewards > .5
    return success.any()


def get_human_data(user_df, user_data, fn, **kwargs):
    eval_df = user_df.filter(**kwargs)
    idxs = np.array(eval_df['index'])-1
    array = []
    for idx in idxs:
        val = fn(user_data[idx])
        array.append(val)

    return array


def get_model_data(model_df, model_data, fn, **kwargs):
    eval_df = model_df.filter(**kwargs)
    idxs = np.array(eval_df['index'])-1
    array = []
    for idx in idxs:
        val = jax.vmap(fn)(model_data[idx])
        array.append(val)

    return np.array(array).mean(-1)
