from typing import Callable
from functools import partial
from flax import struct
import jax
import jax.numpy as jnp
from collections import deque

from agents import value_based_basics as vbb
from housemaze.human_dyna import env as maze
from housemaze import renderer
from projects.humansf import visualizer

import jax.tree_util as jtu
from safetensors.flax import load_file
from flax.traverse_util import unflatten_dict
import pickle


def swap_task(x: maze.TimeStep, w: jax.Array):
    new_state = x.state.replace(
        step_num=jnp.zeros_like(x.state.step_num),
        task_w=w,
    )

    return x.replace(
        state=new_state,
    )
def make_float(x): return x.astype(jnp.float32)

def load_params(filename):
    flattened_dict = load_file(filename)
    return unflatten_dict(flattened_dict, sep=',')

def load_params_config(path: str, file: str, config=True):
    params = load_params(f'{path}/{file}.safetensors')

    if config:
        with open(f'{path}/{file}.config', 'rb') as f:
            config = pickle.load(f)
    return params, config

def collect_trajectory(
    init_timestep,
    task_w,
    rng,
    train_state: vbb.CustomTrainState,
    actor,
    agent_reset_fn,
    vmap_step,
    env_params,
    max_steps: int = 50,
    ):

    rng, rng_ = jax.random.split(rng)
  
    timestep = jax.vmap(swap_task, (0, None))(init_timestep, task_w)
    agent_state = agent_reset_fn(train_state.params, timestep, rng_)

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
                env_params=env_params)

    return traj_batch


def success(timesteps):
  rewards = timesteps.reward
  
  # get mask for within episode
  non_terminal = timesteps.discount
  is_last = timesteps.last()
  term_cumsum = jnp.cumsum(is_last, 0)
  in_episode = make_float((term_cumsum + non_terminal) < 2)

  total_reward = (in_episode*rewards).sum()
  success = make_float((in_episode*rewards).sum() > .5)
  return total_reward, success


def plot_timesteps(
  traj,
  render_fn,
  get_task_name,
  extract_task_info,
  max_len=40,
  action_names=None):
  timesteps = traj.timestep
  actions = traj.action
  #################
  # frames
  #################
  obs_images = []
  for idx in range(max_len):
      index = lambda y: jax.tree_map(lambda x: x[idx], y)
      obs_image = render_fn(index(timesteps.state))
      obs_images.append(obs_image)

  #################
  # actions
  #################
  def action_name(a):
    if action_names is not None:
      name = action_names.get(int(a), 'ERROR?')
      return f"action {int(a)}: {name}"
    else:
      return f"action: {int(a)}"
  actions_taken = [action_name(a) for a in actions]

  #################
  # plot
  #################
  index = lambda t, idx: jax.tree_map(lambda x: x[idx], t)
  def panel_title_fn(timesteps, i):
    task_name = get_task_name(extract_task_info(index(timesteps, i)))
    title = f'{task_name}'

    step_type = int(timesteps.step_type[i])
    step_type = ['first', 'mid', '|last|'][step_type]
    title += f'\nt={i}, type={step_type}'

    if i < len(actions_taken):
      title += f'\n{actions_taken[i]}'
    title += f'\nr={timesteps.reward[i]}, $\\gamma={timesteps.discount[i]}$'

    return title

  fig = visualizer.plot_frames(
      timesteps=timesteps,
      frames=obs_images,
      panel_title_fn=panel_title_fn,
      ncols=6)

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

  def setting(self):
     return self.path.split('/')[-1]


def load_algorithm(
      path,
      name,
      env_params,
      env,
      make_fns,
      parallel_envs: int=25,
      ):
  agent_params, config = load_params_config(path, name)

  config['NUM_ENVS'] = parallel_envs

  def vmap_reset(rng, env_params):
    return jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, config["NUM_ENVS"]), env_params)

  def vmap_step(rng, env_state, action, env_params):
      return jax.vmap(
          env.step, in_axes=(0, 0, 0, None))(
          jax.random.split(rng, config["NUM_ENVS"]), env_state, action, env_params)

  rng = jax.random.PRNGKey(config["SEED"])
  rng, rng_ = jax.random.split(rng)
  example_timestep = vmap_reset(rng_, env_params)

  fns = make_fns(config=config)

  network, _, reset_fn = fns.make_agent(
      config=config,
      env=env,
      env_params=env_params,
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

  def collect_trajectories(
        rng,
        env_params,
        task_w,
        n=4):

      collect_fn = partial(
        collect_trajectory,
        train_state=train_state,
        agent_reset_fn=reset_fn,
        actor=actor,
        vmap_step=vmap_step,
        env_params=env_params,
        )
      collect_fn = jax.vmap(collect_fn, (None, 0, 0))

      def scan_body(rng, _):
          rng, rng_ = jax.random.split(rng)
          init_timestep = vmap_reset(rng=rng_, env_params=env_params)
          rng, rng_ = jax.random.split(rng)
          new_trajs = collect_fn(
            init_timestep, task_w, jax.random.split(rng_, len(task_w)))
          return rng, new_trajs

      # [n, num_tasks, num_timesteps, num_traj, data]
      rng, all_trajs = jax.lax.scan(scan_body, rng, None, length=n)

      def fix_shape(x):
        # [num_tasks, n, num_timesteps, num_traj, data]
        x = jnp.swapaxes(x, 0, 1)

        # [num_tasks, n, num_traj, num_timesteps, data]
        x = jnp.swapaxes(x, 2, 3)

        # [num_tasks, n*num_traj, num_timesteps, data]
        x = x.reshape(x.shape[0], -1, *x.shape[3:])
        return x

      # [n, num_traj, num_timesteps, num_tasks, data]
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
