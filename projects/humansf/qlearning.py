"""
Recurrent Q-learning.
"""



from singleagent import qlearning
import functools
import os
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import NamedTuple, Dict, Union, Optional, Tuple, Callable


import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import flashbax as fbx
import wandb
from math import ceil

import flax
import rlax
from omegaconf import OmegaConf
from safetensors.flax import save_file
from flax.traverse_util import flatten_dict
from gymnax.environments import environment
import matplotlib.pyplot as plt

from xminigrid.rendering.rgb_render import render as rgb_render
from singleagent.basics import TimeStep
from singleagent import value_based_basics as vbb
from projects.humansf.networks import KeyroomObsEncoder
from projects.humansf import keyroom

from library import loggers

Agent = nn.Module
Params = flax.core.FrozenDict
AgentState = flax.struct.PyTreeNode
RNNInput = vbb.RNNInput
R2D2LossFn = qlearning.R2D2LossFn
Predictions = qlearning.Predictions
EpsilonGreedy = qlearning.LinearDecayEpsilonGreedy
make_optimizer = qlearning.make_optimizer
make_loss_fn_class = qlearning.make_loss_fn_class
make_actor = qlearning.make_actor

def extract_timestep_input(timestep: TimeStep):
  return RNNInput(
      obs=timestep.observation,
      reset=timestep.first())

class AgentRNN(nn.Module):
    """_summary_

    - observation encoder: CNN
    Args:
        nn (_type_): _description_

    Returns:
        _type_: _description_
    """
    action_dim: int
    hidden_dim: int
    cell_type: str = "LSTMCell"

    def setup(self):

        self.observation_encoder = KeyroomObsEncoder(self.hidden_dim)

        self.rnn = vbb.ScannedRNN(
           hidden_dim=self.hidden_dim,
           cell_type=self.cell_type)

        self.q_fn = qlearning.MLP(
           hidden_dim=512,
           num_layers=1,
           out_dim=self.action_dim)

    def initialize(self, x: TimeStep):
        """Only used for initialization."""

        rng = jax.random.PRNGKey(0)
        batch_dims = (x.reward.shape[0],)
        rnn_state = self.initialize_carry(rng, batch_dims)

        return self.__call__(rnn_state, x, rng)

    def __call__(self, rnn_state, x: TimeStep, rng: jax.random.KeyArray):
        x = extract_timestep_input(x)

        embedding = self.observation_encoder(x.obs)
        embedding = nn.relu(embedding)

        rnn_in = x._replace(obs=embedding)
        rng, _rng = jax.random.split(rng)
        new_rnn_state, rnn_out = self.rnn(rnn_state, rnn_in, _rng)

        q_vals = self.q_fn(rnn_out)

        return Predictions(q_vals, rnn_out), new_rnn_state

    def unroll(self, rnn_state, xs: TimeStep, rng: jax.random.KeyArray):
        # rnn_state: [B]
        # xs: [T, B]
        xs = extract_timestep_input(xs)

        embedding = nn.BatchApply(self.observation_encoder)(xs.obs)
        embedding = nn.relu(embedding)

        rnn_in = xs._replace(obs=embedding)
        rng, _rng = jax.random.split(rng)
        new_rnn_state, rnn_out = self.rnn.unroll(rnn_state, rnn_in, _rng)

        q_vals = nn.BatchApply(self.q_fn)(rnn_out)

        return Predictions(q_vals, rnn_out), new_rnn_state

    def initialize_carry(self, *args, **kwargs):
        """Initializes the RNN state."""
        return self.rnn.initialize_carry(*args, **kwargs)

def make_agent(
        config: dict,
        env: environment.Environment,
        env_params: environment.EnvParams,
        example_timestep: TimeStep,
        rng: jax.random.KeyArray) -> Tuple[Agent, Params, vbb.AgentResetFn]:

    agent = AgentRNN(
        action_dim=env.num_actions(env_params),
        hidden_dim=config["AGENT_HIDDEN_DIM"],
    )

    rng, _rng = jax.random.split(rng)
    network_params = agent.init(
        _rng, example_timestep, method=agent.initialize)

    def reset_fn(params, example_timestep, reset_rng):
      batch_dims = (example_timestep.reward.shape[0],)
      return agent.apply(
          params,
          batch_dims=batch_dims,
          rng=reset_rng,
          method=agent.initialize_carry)

    return agent, network_params, reset_fn


def plot_frames(task_name, frames, rewards, actions_taken, W, max_frames=1e10):
    """
    Dynamically plots frames in a single figure based on the number of columns (W) and maximum number of frames.
    
    :param task_name: Name of the task to be displayed as figure title.
    :param frames: 4D numpy array of shape (T, H, W, C) containing the frames to plot.
    :param rewards: List of rewards corresponding to actions taken.
    :param actions_taken: List of actions taken corresponding to each frame.
    :param max_frames: Maximum number of frames to plot.
    :param W: Number of columns in the plot's grid.
    """
    T = min(len(frames), max_frames)  # Total number of frames to plot (limited by max_frames)
    H = ceil(T / W)  # Calculate number of rows required
    width = 3
    
    fig, axs = plt.subplots(H, W, figsize=(W*width, H*width), squeeze=False)
    fig.suptitle(task_name)
    
    # Flatten the axes array for easy iteration
    axs = axs.ravel()
    
    for i in range(T):
        ax = axs[i]
        ax.imshow(frames[i])
        ax.axis('off')  # Hide the axis

        if i < len(actions_taken) and i < len(rewards):
            ax.set_title(f"{actions_taken[i]}, r={rewards[i]}")

    # Hide unused subplots
    for i in range(T, H * W):
        axs[i].axis('off')

    plt.tight_layout()

    return fig

def get_task_name(room_setting: int, task_room: int, task_object: int, maze_config: dict):
    setting = 'single' if room_setting == 0 else 'multi'
    category, color = maze_config['pairs'][task_room][task_object]
    return f'{setting} - {color} {category}'

def make_logger(
        config: dict,
        env: environment.Environment,
        env_params: environment.EnvParams,
        maze_config: dict):


    def qlearner_logger(data: dict):
        def callback(d):
            n_updates = d.pop('n_updates')

            # Extract the relevant data
            # only use data from batch dim = 0
            # [T, B, ...] --> # [T, ...]
            d_ = jax.tree_map(lambda x: x[:, 0], d)

            rewards = d_['data'].timestep.reward[1:]
            actions = d_['data'].action[:-1]
            q_values = d_['q_values'][:-1]
            q_values_taken = np.take_along_axis(
                q_values, actions[..., None], axis=-1).squeeze(-1)
            td_errors = d_['td_errors']
            q_loss = d_['q_loss']

            # Create a figure with three subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))

            # Plot rewards and q-values in the top subplot
            ax1.plot(rewards, label='Rewards')
            ax1.plot(q_values_taken, label='Q-Values')
            ax1.set_xlabel('Time')
            ax1.set_title('Rewards and Q-Values')
            ax1.legend()

            # Plot TD errors in the middle subplot
            ax2.plot(td_errors)
            ax2.set_xlabel('Time')
            ax2.set_title('TD Errors')

            # Plot Q-loss in the bottom subplot
            ax3.plot(q_loss)
            ax3.set_xlabel('Update')
            ax3.set_title('Q-Loss')

            # Adjust the spacing between subplots
            plt.tight_layout()
            # log
            if wandb.run is not None:
                wandb.log({f"learner_example/q-values": wandb.Image(fig)})
            plt.close(fig)

            ##############################
            # plot images of env
            ##############################
            action_names = {
                0: 'left',
                1: 'right',
                2: 'forward',
                3: 'pickup',
                4: 'drop',
                5: 'toggle',
                6: 'done'}

            #timestep = jax.tree_map(lambda x: jnp.array(x), d_['data'].timestep)
            timestep = d_['data'].timestep

            # ------------
            # get images
            # ------------

            state_images = []
            obs_images = []
            max_len = min(config.get("MAX_EPISODE_LOG_LEN", 40), len(rewards))
            for idx in range(max_len):
                index = lambda y: jax.tree_map(lambda x: x[idx], y)
                state_image = rgb_render(
                    timestep.state.grid[idx],
                    index(timestep.state.agent),
                    env_params.view_size,
                    tile_size=8)
                obs_image = keyroom.render_room(
                    index(d_['data'].timestep.state),
                    tile_size=8)
                state_images.append(state_image)
                obs_images.append(obs_image)

            # ------------
            # task name
            # ------------
            room_setting = int(timestep.state.room_setting[0])
            task_room = int(timestep.state.goal_room_idx[0])
            task_object = int(timestep.state.task_object_idx[0])
            setting = 'single' if room_setting == 0 else 'multi'
            category, color = maze_config['pairs'][task_room][task_object]
            task_name = f'{setting} - {color} {category}'

            # ------------
            # plot
            # ------------
            actions_taken = [action_names[int(a)] for a in d_['data'].action]
            fig = plot_frames(task_name,
                        frames=obs_images,
                        rewards=rewards,
                        actions_taken=actions_taken,
                        W=10)
            if wandb.run is not None:
                wandb.log({f"learner_example/trajecotry": wandb.Image(fig)})
            plt.close(fig)


        jax.lax.cond(
            data['n_updates'] % config.get("LEARNER_LOG_PERIOD", 10_000) == 0,
            lambda d: jax.debug.callback(callback, d),
            lambda d: None,
            data)

    return loggers.Logger(
        gradient_logger=loggers.default_gradient_logger,
        learner_logger=loggers.default_learner_logger,
        experience_logger=loggers.default_experience_logger,
        learner_log_extra=qlearner_logger,
    )
