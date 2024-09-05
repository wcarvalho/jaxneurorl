"""
Recurrent Q-learning.
"""



import os
import jax
from typing import Tuple, Callable


import flax.linen as nn
import flashbax as fbx
import wandb

import flax
import rlax
from gymnax.environments import environment
import matplotlib.pyplot as plt

from xminigrid.rendering.rgb_render import render as rgb_render


from projects.humansf.networks import KeyroomObsEncoder, HouzemazeObsEncoder
from projects.humansf import keyroom
from projects.humansf.visualizer import plot_frames

from jaxneurorl.agents.basics import TimeStep
from jaxneurorl.agents import value_based_basics as vbb
from jaxneurorl.agents import qlearning as base_agent



Agent = nn.Module
Params = flax.core.FrozenDict
AgentState = flax.struct.PyTreeNode
RNNInput = vbb.RNNInput
R2D2LossFn = base_agent.R2D2LossFn
Predictions = base_agent.Predictions
make_optimizer = base_agent.make_optimizer
make_loss_fn_class = base_agent.make_loss_fn_class
make_actor = base_agent.make_actor
epsilon_greedy_act = base_agent.epsilon_greedy_act

class RnnAgent(nn.Module):
    """_summary_

    - observation encoder: CNN
    Args:
        nn (_type_): _description_

    Returns:
        _type_: _description_
    """
    action_dim: int

    observation_encoder: nn.Module
    rnn: vbb.ScannedRNN

    def setup(self):

        self.q_fn = base_agent.MLP(
           hidden_dim=512,
           num_layers=1,
           out_dim=self.action_dim)

    def initialize(self, x: TimeStep):
        """Only used for initialization."""

        rng = jax.random.PRNGKey(0)
        batch_dims = (x.reward.shape[0],)
        rnn_state = self.initialize_carry(rng, batch_dims)

        return self(rnn_state, x, rng)

    def __call__(self, rnn_state, x: TimeStep, rng: jax.random.PRNGKey):

        embedding = self.observation_encoder(x.observation)

        rnn_in = RNNInput(obs=embedding, reset=x.first())
        rng, _rng = jax.random.split(rng)
        new_rnn_state, rnn_out = self.rnn(rnn_state, rnn_in, _rng)

        q_vals = self.q_fn(rnn_out)

        return Predictions(q_vals, rnn_out), new_rnn_state

    def unroll(self, rnn_state, xs: TimeStep, rng: jax.random.PRNGKey):
        # rnn_state: [B]
        # xs: [T, B]

        embedding = nn.BatchApply(self.observation_encoder)(xs.observation)

        rnn_in = RNNInput(obs=embedding, reset=xs.first())
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
        rng: jax.random.PRNGKey,
        ObsEncoderCls: nn.Module = KeyroomObsEncoder,
        ) -> Tuple[Agent, Params, vbb.AgentResetFn]:

    cell_type = config.get('RNN_CELL_TYPE', 'OptimizedLSTMCell')
    if cell_type.lower() == 'none':
        rnn = vbb.DummyRNN()
    else:
        rnn = vbb.ScannedRNN(
            hidden_dim=config["AGENT_RNN_DIM"],
            cell_type=cell_type,
            )

    agent = RnnAgent(
        observation_encoder=ObsEncoderCls(),
        action_dim=env.num_actions(env_params),
        rnn=rnn,
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

def learner_log_extra(
        data: dict,
        config: dict,
        action_names: dict,
        render_fn: Callable,
        extract_task_info: Callable[[TimeStep],
                                    flax.struct.PyTreeNode] = lambda t: t,
        get_task_name: Callable = lambda t: 'Task',
        ):
    def callback(d):
        n_updates = d.pop('n_updates')

        # Extract the relevant data
        # only use data from batch dim = 0
        # [T, B, ...] --> # [T, ...]
        d_ = jax.tree_map(lambda x: x[:, 0], d)

        mask = d_['mask']
        discounts = d_['data'].timestep.discount
        rewards = d_['data'].timestep.reward
        actions = d_['data'].action
        q_values = d_['q_values']
        q_target = d_['q_target']
        q_values_taken = rlax.batched_index(q_values, actions)
        td_errors = d_['td_errors']
        q_loss = d_['q_loss']
        # Create a figure with three subplots
        width = .3
        nT = len(rewards)  # e.g. 20 --> 8
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(int(width*nT), 16))

        # Plot rewards and q-values in the top subplot
        def format(ax):
            ax.set_xlabel('Time')
            ax.grid(True)
            ax.set_xticks(range(0, len(rewards), 1))
        ax1.plot(rewards, label='Rewards')
        ax1.plot(q_values_taken, label='Q-Values')
        ax1.plot(q_target, label='Q-Targets')
        ax1.set_title('Rewards and Q-Values')
        format(ax1)
        ax1.legend()

        # Plot TD errors in the middle subplot
        ax2.plot(td_errors)
        format(ax2)
        ax2.set_title('TD Errors')

        # Plot Q-loss in the bottom subplot
        ax3.plot(q_loss)
        format(ax3)
        ax3.set_title('Q-Loss')

        # Plot episode quantities
        is_last = d_['data'].timestep.last()
        ax4.plot(discounts, label='Discounts')
        ax4.plot(mask, label='mask')
        ax4.plot(is_last, label='is_last')
        format(ax4)
        ax4.set_title('Episode markers')
        ax4.legend()

        # Adjust the spacing between subplots
        #plt.tight_layout()
        # log
        if wandb.run is not wandb.sdk.lib.disabled.RunDisabled:
            wandb.log({f"learner_example/q-values": wandb.Image(fig)})
        plt.close(fig)

        ##############################
        # plot images of env
        ##############################
        #timestep = jax.tree_map(lambda x: jnp.array(x), d_['data'].timestep)
        timesteps: TimeStep = d_['data'].timestep

        # ------------
        # get images
        # ------------

        #state_images = []
        obs_images = []
        max_len = min(config.get("MAX_EPISODE_LOG_LEN", 40), len(rewards))
        for idx in range(max_len):
            index = lambda y: jax.tree_map(lambda x: x[idx], y)
            #state_image = rgb_render(
            #    timesteps.state.grid[idx],
            #    index(timesteps.state.agent),
            #    env_params.view_size,
            #    tile_size=8)
            obs_image = render_fn(index(d_['data'].timestep.state))
            #state_images.append(state_image)
            obs_images.append(obs_image)

        # ------------
        # plot
        # ------------
        def action_name(a):
            if action_names is not None:
                name = action_names.get(int(a), 'ERROR?')
                return f"action {int(a)}: {name}"
            else:
                return f"action: {int(a)}"
        actions_taken = [action_name(a) for a in actions]

        def index(t, idx): return jax.tree_map(lambda x: x[idx], t)
        def panel_title_fn(timesteps, i):
            #room_setting = int(timesteps.state.room_setting[i])
            #task_room = int(timesteps.state.goal_room_idx[i])
            #task_object = int(timesteps.state.task_object_idx[i])
            #setting = 'single' if room_setting == 0 else 'multi'
            #category, color = maze_config['pairs'][task_room][task_object]
            #task_name = f'{setting} - {color} {category}'
            task_name = get_task_name(extract_task_info(index(timesteps, i)))
            title = f'{task_name}\n'
            title += f't={i}\n'
            title += f'{actions_taken[i]}\n'
            title += f'r={timesteps.reward[i]}, $\\gamma={timesteps.discount[i]}$'
            return title

        fig = plot_frames(
            timesteps=timesteps,
            frames=obs_images,
            panel_title_fn=panel_title_fn,
            ncols=6)
        if wandb.run is not wandb.sdk.lib.disabled.RunDisabled:
            wandb.log(
                {f"learner_example/trajecotry": wandb.Image(fig)})
        plt.close(fig)

    # this will be the value after update is applied
    n_updates = data['n_updates'] + 1
    is_log_time = n_updates % config["LEARNER_EXTRA_LOG_PERIOD"] == 0

    jax.lax.cond(
        is_log_time,
        lambda d: jax.debug.callback(callback, d),
        lambda d: None,
        data)
