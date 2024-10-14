"""
Universal Successor Feature Approximator (USFA)
"""

import jax
from typing import Callable


import wandb

import flax
import rlax
import matplotlib.pyplot as plt

from projects.humansf.visualizer import plot_frames

from jaxneurorl.agents.basics import TimeStep
from jaxneurorl.agents.usfa import *

# only redoing this
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
        cumulants = d_['cumulants']  # [T, C]
        sf_values = d_['sf_values'][:, 0]  # [T, A, C]
        sf_target = d_['sf_target'][:, 0]  # [T, C]
        # [T, C]
        # vmap over cumulant dimension
        sf_values_taken = jax.vmap(rlax.batched_index, in_axes=(
            2, None), out_axes=1)(sf_values, actions[:-1])
        #sf_td_errors = d_['td_errors']  # [T, C]
        #sf_loss = d_['sf_loss']  # [T, C]

        ##############################
        # SF-learning plots
        ##############################
        # Create a figure with subplots for each cumulant
        num_cumulants = cumulants.shape[-1]
        num_cols = min(4, num_cumulants)
        num_rows = (num_cumulants + num_cols - 1) // num_cols
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
        axes = axes.flatten()

        for i in range(num_cumulants):
            ax = axes[i]
            time_steps = range(len(cumulants))

            ax.plot(time_steps, cumulants[:, i], label='Cumulants')
            ax.plot(time_steps, sf_values_taken[:, i], label='SF Values Taken')
            ax.plot(time_steps, sf_target[:, i], label='SF Target')
            ax.set_title(f'Cumulant {i+1}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)

        # Remove any unused subplots
        for i in range(num_cumulants, len(axes)):
            fig.delaxes(axes[i])

        #plt.tight_layout()

        # Log the Q-learning figure
        if wandb.run is not wandb.sdk.lib.disabled.RunDisabled:
            wandb.log({f"learner_example/sf-learning": wandb.Image(fig)})
        plt.close(fig)

        ##############################
        # Q-learning plots
        ##############################
        # Plot rewards and q-values in the top subplot
        width = .3
        nT = len(rewards)  # e.g. 20 --> 8

        task = d_['data'].timestep.observation.task_w[:-1]
        q_values_taken = (sf_values_taken*task).sum(-1)
        q_target = (sf_target*task).sum(-1)
        td_errors = jnp.abs(q_target - q_values_taken)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(int(width*nT), 16))
        def format(ax):
            ax.set_xlabel('Time')
            ax.grid(True)
            ax.set_xticks(range(0, len(rewards), 1))
        # Set the same x-limit for all subplots
        x_max = len(rewards)
        
        ax1.plot(rewards, label='Rewards')
        ax1.plot(q_values_taken, label='Q-Values')
        ax1.plot(q_target, label='Q-Targets')
        ax1.set_title('Rewards and Q-Values')
        ax1.set_xlim(0, x_max)
        format(ax1)
        ax1.legend()

        # Plot TD errors in the middle subplot
        ax2.plot(td_errors)
        ax2.set_xlim(0, x_max)
        format(ax2)
        ax2.set_title('TD Errors')

        # Plot episode quantities
        is_last = d_['data'].timestep.last()
        ax3.plot(discounts, label='Discounts')
        ax3.plot(mask, label='mask')
        ax3.plot(is_last, label='is_last')
        ax3.set_xlim(0, x_max)
        format(ax3)
        ax3.set_title('Episode markers')
        ax3.legend()

        # Ensure all subplots have the same x-axis range
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)

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
