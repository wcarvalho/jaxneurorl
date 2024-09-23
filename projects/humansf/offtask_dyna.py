"""
Dyna with the ability to do off-task simulation.
"""

from typing import Tuple, Optional, Union, Callable
import functools

import distrax
import jax
import jax.numpy as jnp
import flax
from flax import struct
import optax
import flax.linen as nn
from gymnax.environments import environment
import numpy as np
import rlax
import matplotlib.pyplot as plt
import wandb

from jaxneurorl import losses
from jaxneurorl.agents.basics import TimeStep
from jaxneurorl.agents import value_based_basics as vbb
from jaxneurorl.agents import qlearning as base_agent
from projects.humansf import qlearning
from projects.humansf.networks import KeyroomObsEncoder, MLP
from projects.humansf import keyroom
from projects.humansf.visualizer import plot_frames

from housemaze import renderer

Agent = nn.Module
Params = flax.core.FrozenDict
Qvalues = jax.Array
RngKey = jax.Array
make_actor = base_agent.make_actor

RnnState = jax.Array
SimPolicy = Callable[[Qvalues, RngKey], int]


@struct.dataclass
class AgentState:
    timestep: jax.Array
    rnn_state: jax.Array

@struct.dataclass
class Predictions:
    q_vals: jax.Array
    state: AgentState


@struct.dataclass
class SimulationOutput:
    actions: jax.Array
    predictions: Predictions


def make_float(x): return x.astype(jnp.float32)

def concat_pytrees(tree1, tree2, **kwargs):
    return jax.tree_map(lambda x, y: jnp.concatenate((x, y), **kwargs), tree1, tree2)

def add_time(v): return jax.tree_map(lambda x: x[None], v)
def concat_first_rest(first, rest):
    # init: [N, ...]
    # rest: [T, N, ...]
    # output: [T+1, N, ...]
    return jax.vmap(concat_pytrees, 1, 1)(add_time(first), rest)



def make_optimizer(config: dict) -> optax.GradientTransformation:
  num_updates = int(config["NUM_UPDATES"] + config.get("NUM_EXTRA_REPLAY", 0))

  lr_scheduler = optax.linear_schedule(
      init_value=config["LR"],
      end_value=1e-10,
      transition_steps=num_updates)

  lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

  return optax.chain(
      optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
      optax.adam(learning_rate=lr, eps=config['EPS_ADAM'])
  )


def  simulate_n_trajectories(
        x_t: TimeStep,
        h_tm1: RnnState,
        rng:jax.random.PRNGKey,
        network: nn.Module,
        params: Params,
        #temperatures: Optional[jax.Array] = None,
        policy_fn: SimPolicy = None,
        num_steps: int = 5,
        num_simulations: int = 5,
    ):
    """

    return predictions and actions for every time-step including the current one.

    This first applies the model to the current time-step and then simulates T more time-steps. 
    Output is num_steps+1.

    Args:
        x_t (TimeStep): [D]
        h_tm1 (RnnState): [D]
        rng (jax.random.PRNGKey): _description_
        network (nn.Module): _description_
        params (Params): _description_
        temperatures (jax.Array): _description_
        num_steps (int, optional): _description_. Defaults to 5.
        num_simulations (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    """
    #if temperatures is None:
    #   temperatures = jnp.ones(num_simulations)
    #assert len(temperatures) == num_simulations

    #def policy_fn(q_values, temp, rng_):
    #   # q_values: [N, A] or [A]
    #   # temp: [N] or []
    #   logits = q_values / jnp.expand_dims(temp, -1)
    #   return distrax.Categorical(
    #       logits=logits).sample(seed=rng_)

    def initial_predictions(x, prior_s, rng_):
        preds, s = network.apply(params, prior_s, x, rng_)
        return preds

    # by giving state as input and returning, will
    # return copies. 1 for each sampled action.
    rng, rng_ = jax.random.split(rng)

    # one for each simulation
    # [N, ...]
    init_preds_t = jax.vmap(
        initial_predictions,
        in_axes=(None, None, 0),
        out_axes=(0)
        )(x_t,           # [D]
          h_tm1,         # [D]
          jax.random.split(rng_, num_simulations)
        )
    init_a_t = policy_fn(
        init_preds_t, rng_)

    def _single_model_step(carry, inputs):
        del inputs  # unused
        (s, a, rng) = carry

        ###########################
        # 1. use state + action to predict next state
        ###########################
        rng, rng_ = jax.random.split(rng)
        next_preds, _ = network.apply(
            params, s, a, rng_,
            method=network.apply_model)

        ###########################
        # 2. get actions at next state
        ###########################
        # [N]
        next_a = policy_fn(
            next_preds,
            rng_,
        )
        carry = (next_preds.state, next_a, rng)
        sim_output = SimulationOutput(
            predictions=next_preds,
            actions=next_a,
        )
        return carry, sim_output

    ################
    # get simulation ouputs
    ################
    initial_carry = (init_preds_t.state, init_a_t, rng)
    _, sim_outputs = jax.lax.scan(
        f=_single_model_step,
        init=initial_carry,
        xs=None, length=num_steps)

    # sim_outputs.predictions: [T, N, ...]
    # concat [1, ...] with [N, T, ...]
    return SimulationOutput(
        predictions=concat_first_rest(
            init_preds_t, sim_outputs.predictions),
        actions=concat_first_rest(init_a_t, sim_outputs.actions),
    )


@struct.dataclass
class OfftaskDyna(vbb.RecurrentLossFn):

    """Loss function for off-task dyna.

    Note: this assumes the agent uses the ground-truth environment as the environment model.
    """
    num_simulations: int = 15
    simulation_length: int = 5
    online_coeff: float = 1.0
    dyna_coeff: float = 1.0

    offtask_simulation: bool = True
    stop_dyna_gradient: bool = True

    env_params: environment.EnvParams = None

    make_init_offtask_timestep: Callable[[TimeStep], TimeStep] = None
    simulation_policy: SimPolicy = None
    temp_dist: distrax.Distribution = distrax.Gamma(concentration=1, rate=.5)

    def loss_fn(
        self,
        timestep,
        online_preds,
        target_preds,
        actions,
        rewards,
        is_last,
        non_terminal,
        loss_mask,
        ):

        rewards = make_float(rewards)
        is_last = make_float(is_last)
        discounts = make_float(non_terminal)*self.discount
        lambda_ = jnp.ones_like(non_terminal)*self.lambda_

        # Get N-step transformed TD error and loss.
        batch_td_error_fn = jax.vmap(
            losses.q_learning_lambda_td,
            in_axes=1,
            out_axes=1)

        # [T, B]
        selector_actions = jnp.argmax(online_preds.q_vals, axis=-1)  # [T+1, B]
        q_t, target_q_t = batch_td_error_fn(
            online_preds.q_vals[:-1],  # [T+1] --> [T]
            actions[:-1],    # [T+1] --> [T]
            target_preds.q_vals[1:],  # [T+1] --> [T]
            selector_actions[1:],  # [T+1] --> [T]
            rewards[1:],        # [T+1] --> [T]
            discounts[1:],
            is_last[1:],
            lambda_[1:])      # [T+1] --> [T]

        # ensure target = 0 when episode terminates
        target_q_t = target_q_t*non_terminal[:-1]
        batch_td_error = target_q_t - q_t
        batch_td_error = batch_td_error*loss_mask[:-1]

        # [T, B]
        batch_loss = 0.5 * jnp.square(batch_td_error)

        # [B]
        batch_loss_mean = (batch_loss*loss_mask[:-1]).mean(0)

        metrics = {
            '0.q_loss': batch_loss.mean(),
            '0.q_td': jnp.abs(batch_td_error).mean(),
            '1.reward': rewards[1:].mean(),
            'z.q_mean': online_preds.q_vals.mean(),
            'z.q_var': online_preds.q_vals.var(),
            }

        log_info = {
            'timesteps': timestep,
            'actions': actions,
            'td_errors': batch_td_error,                 # [T]
            'loss_mask': loss_mask,                 # [T]
            'q_values': online_preds.q_vals,    # [T, B]
            'q_loss': batch_loss,                        #[ T, B]
            'q_target': target_q_t,
        }

        return batch_td_error, batch_loss_mean, metrics, log_info

    def error(self,
        data,
        online_preds: Predictions,
        online_state: AgentState,
        target_preds: Predictions,
        target_state: AgentState,
        params: Params,
        target_params: Params,
        steps: int,
        key_grad: jax.random.PRNGKey,
        **kwargs):

        ##################
        ## Q-learning loss on batch of data
        ##################

        # prepare data
        non_terminal = data.timestep.discount
        # either termination or truncation
        is_last = make_float(data.timestep.last())

        # truncated is discount on AND is last
        truncated = (non_terminal+is_last) > 1
        loss_mask = make_float(1-truncated)

        all_metrics = {}
        all_log_info = {
            'n_updates': steps,
        }
        if self.online_coeff > 0.0:
            td_error, batch_loss, metrics, log_info = self.loss_fn(
                timestep=data.timestep,
                online_preds=online_preds,
                target_preds=target_preds,
                actions=data.action,
                rewards=data.reward,
                is_last=is_last,
                non_terminal=non_terminal,
                loss_mask=loss_mask,
                )

            # first label online loss with online
            all_metrics.update({f'online/{k}': v for k, v in metrics.items()})
            all_log_info['online'] = log_info
        else:
            td_error = jnp.zeros_like((loss_mask[:-1]))
            batch_loss = td_error.sum(0)  # time axis

        #################
        # Dyna Q-learning loss over simulated data
        #################
        if self.dyna_coeff > 0.0:
            if self.stop_dyna_gradient:
                sg = lambda x: jax.lax.stop_gradient(x)
            else:
                sg = lambda x: x
            # get states at t=1, ....
            # tm1 = t-1
            # [T-1, B, ...]
            h_tm1_online = jax.tree_map(
                lambda x: sg(x[:-1]), online_preds.state.rnn_state)
            h_tm1_target = jax.tree_map(
                lambda x: sg(x[:-1]), target_preds.state.rnn_state)

            # get time-steps at t=2, ...
            # [T-1, B, ...]
            x_t = jax.tree_map(
                lambda x: sg(x[1:]), online_preds.state.timestep)

            if self.offtask_simulation:
                assert self.make_init_offtask_timestep is not None
                #--------------
                # get off task goal and place in timestep
                # --------------
                # [T-1, B, ...]
                # for now, just a single off-task goal
                # TODO: generalize to doing this for multiple tasks
                # TODO: right now, rely on task being part of state. next step should not be.
                # TODO: generalize this process for other environments
                offtask_w = x_t.state.offtask_w
                x_t = self.make_init_offtask_timestep(x_t, offtask_w)

            T, B = offtask_w.shape[:2]
            rngs = jax.random.split(key_grad, T*B)
            rngs = rngs.reshape(T, B, -1)

            dyna_loss_fn = functools.partial(
                self.dyna_loss_fn,
                params=params,
                target_params=target_params)

            # vmap over batch + time
            dyna_loss_fn = jax.vmap(jax.vmap(dyna_loss_fn))
            dyna_td_error, dyna_batch_loss, dyna_metrics, dyna_log_info = dyna_loss_fn(
                        x_t,
                        h_tm1_online,
                        h_tm1_target,
                        rngs,
                    )
            # [time, batch, num_sim, sim_length]
            # average over (num_sim, sim_length)
            dyna_td_error = dyna_td_error.mean(axis=(2, 3))
            # average over (time, num_sim)
            dyna_batch_loss = dyna_batch_loss.mean(axis=(0, 2))

            td_error += self.dyna_coeff*dyna_td_error

            batch_loss += self.dyna_coeff*dyna_batch_loss

            # update metrics with dyna metrics
            all_metrics.update(
                {f'dyna/{k}': v for k, v in dyna_metrics.items()})

            all_log_info['dyna'] = dyna_log_info

        if self.logger.learner_log_extra is not None:
            self.logger.learner_log_extra(all_log_info)
        
        return td_error, batch_loss, all_metrics

    def dyna_loss_fn(
        self,
        x_t: TimeStep,
        h_tm1_online: jax.Array,
        h_tm1_target: jax.Array,
        rng: jax.random.PRNGKey,
        params,
        target_params,
        ):
        """

        Algorithm:
        - Sample G (off-task) goals
        - for each goal, sample N trajectories
        - compute loss over trajectories

        Details:
        - for each

        Args:
            x_t (TimeStep): [D], timestep at t
            h_tm1 (jax.Array): [D], rnn-state at t-1
            h_tm1_target (jax.Array): [D], rnn-state at t-1 from target network
        """

        ################
        # Sample trajectories
        ################
        # what do you need to simulate a trajectory?
        # need a network with model, history fn, etc.
        # need an initial agent state
        # need an rng
        # will first simulate trajectory using online parameters
        # input = (a) network (b) initial state (c) rng
        # output = (a) online_states (b) online_preds (c) actions
        # GIVEN: s_0, simulation_length=2
        # OUTPUT: a_0, s_1, a_1, s_2
        #   also: online_preds

        #rng, rng_ = jax.random.split(rng)
        #temperatures = self.temp_dist.sample(
        #    seed=rng_,
        #    sample_shape=(self.num_simulations,))

        rng, rng_ = jax.random.split(rng)
        # [num_sim, ...]
        sim_outputs_t = simulate_n_trajectories(
            x_t=x_t,
            h_tm1=h_tm1_online,
            rng=rng_,
            network=self.network,
            params=params,
            num_steps=self.simulation_length,
            num_simulations=self.num_simulations,
            policy_fn=self.simulation_policy,
        )
        preds_t_online = sim_outputs_t.predictions

        ################################
        # function to copy something n times
        ################################
        def identity(y, n): return y
        def repeat_num_sim(x):
            return jax.vmap(
                identity, (None, 0), 0)(
                    x, jnp.arange(self.num_simulations))

        ################################
        # get target_preds for this data
        ################################
        # prediction for t=1
        rng, rng_ = jax.random.split(rng)

        # [D]
        first_preds_target, _ = self.network.apply(
            target_params, h_tm1_target, x_t, rng_)

        # [N, D]
        init_state = repeat_num_sim(first_preds_target.state)

        # [T, N]
        simulation_actions = sim_outputs_t.actions[:-1]
        # prediction for t=2,...
        # [T+1, N, ...]
        _, rest_preds_target = self.network.apply(
            target_params,
            init_state,
            simulation_actions,
            rng_,
            method=self.network.unroll_model)

        # [1, D]
        first_preds_target = add_time(first_preds_target)

        # prediction for t=1,2,...
        # [T+1, N, D]
        preds_t_target = jax.vmap(concat_pytrees, (None, 1), 1)(
            first_preds_target, # [1, D]
            rest_preds_target,  # [T, N, D]
        )

        ################################
        # get timestep information
        ################################
        # agent state stores current state, so should have all of them
        timesteps_t: TimeStep = preds_t_online.state.timestep

        ################
        # Apply loss function to trajectories
        ################
        # prepare data
        non_terminal = timesteps_t.discount
        # either termination or truncation
        is_last_t = make_float(timesteps_t.last())

        # time-step of termination and everything afterwards is masked out
        term_cumsum_t = jnp.cumsum(is_last_t, 0)
        loss_mask_t = make_float((term_cumsum_t + non_terminal) < 2)

        batch_td_error, batch_loss_mean, metrics, log_info = self.loss_fn(
            timestep=timesteps_t,
            online_preds=preds_t_online,
            target_preds=preds_t_target,
            actions=sim_outputs_t.actions,
            rewards=timesteps_t.reward,
            is_last=is_last_t,
            non_terminal=timesteps_t.discount,
            loss_mask=loss_mask_t,
        )

        return batch_td_error, batch_loss_mean, metrics, log_info

def make_loss_fn_class(config, **kwargs) -> vbb.RecurrentLossFn:
    return functools.partial(
        OfftaskDyna,
        discount=config['GAMMA'],
        lambda_=config.get('TD_LAMBDA', .9),
        num_simulations=config.get('NUM_SIMULATIONS', 15),
        simulation_length=config.get('SIMULATION_LENGTH', 5),
        stop_dyna_gradient=config.get('STOP_DYNA_GRAD', True),
        **kwargs
        )

def get_in_episode(timestep):
  # get mask for within episode
  non_terminal = timestep.discount
  is_last = timestep.last()
  term_cumsum = jnp.cumsum(is_last, -1)
  in_episode = (term_cumsum + non_terminal) < 2
  return in_episode

def learner_log_extra(
        data: dict,
        config: dict,
        action_names: dict,
        render_fn: Callable,
        extract_task_info: Callable[[TimeStep],
                                    flax.struct.PyTreeNode] = lambda t: t,
        get_task_name: Callable = lambda t: 'Task',
        sim_idx: int = 0,
        ):

    def log_data(
        key: str, 
        timesteps: TimeStep,
        actions: np.array,
        td_errors: np.array,
        loss_mask: np.array,
        q_values: np.array,
        q_loss: np.array,
        q_target: np.array,
        ):

        # Extract the relevant data
        # only use data from batch dim = 0
        # [T, B, ...] --> # [T, ...]

        discounts = timesteps.discount
        rewards = timesteps.reward
        q_values_taken = rlax.batched_index(q_values, actions)

        # Create a figure with three subplots
        width = .3
        nT = len(rewards)  # e.g. 20 --> 8
        width = max(int(width*nT), 10)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(width, 20))

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
        is_last = timesteps.last()
        ax4.plot(discounts, label='Discounts')
        ax4.plot(loss_mask, label='mask')
        ax4.plot(is_last, label='is_last')
        format(ax4)
        ax4.set_title('Episode markers')
        ax4.legend()

        if wandb.run is not None:
            wandb.log({f"learner_example/{key}/q-values": wandb.Image(fig)})
        plt.close(fig)

        ##############################
        # plot images of env
        ##############################
        # initial image
        maze_height, maze_width, _ = timesteps.state.grid[0].shape
        fig, ax = plt.subplots(1, figsize=(5, 5))
        in_episode = get_in_episode(timesteps)
        actions = actions[in_episode][:-1]
        positions = jax.tree_map(lambda x: x[in_episode][:-1], timesteps.state.agent_pos)

        img = render_fn(jax.tree_map(lambda x: x[0], timesteps.state))
        renderer.place_arrows_on_image(
            img,
            positions,
            actions,
            maze_height, maze_width, arrow_scale=5, ax=ax)
        if wandb.run is not None:
            wandb.log(
                {f"learner_example/{key}/trajectory": wandb.Image(fig)})
        plt.close(fig)
        ## ------------
        ## get images
        ## ------------

        ##state_images = []
        #obs_images = []
        #max_len = min(config.get("MAX_EPISODE_LOG_LEN", 40), len(rewards))
        #for idx in range(max_len):
        #    index = lambda y: jax.tree_map(lambda x: x[idx], y)
        #    #state_image = rgb_render(
        #    #    timesteps.state.grid[idx],
        #    #    index(timesteps.state.agent),
        #    #    env_params.view_size,
        #    #    tile_size=8)
        #    obs_image = render_fn(index(timesteps.state))
        #    #obs_image = keyroom.render_room(
        #    #    index(timesteps.state),
        #    #    tile_size=8)
        #    #state_images.append(state_image)
        #    obs_images.append(obs_image)

        ## ------------
        ## plot
        ## ------------
        #def action_name(a):
        #    if action_names is not None:
        #        name = action_names.get(int(a), 'ERROR?')
        #        return f"action {int(a)}: {name}"
        #    else:
        #        return f"action: {int(a)}"
        #actions_taken = [action_name(a) for a in actions]

        #index = lambda t, idx: jax.tree_map(lambda x: x[idx], t)
        #def panel_title_fn(timesteps, i):
        #    task_name = get_task_name(extract_task_info(index(timesteps, i)))
        #    #room_setting = int(timesteps.state.room_setting[i])
        #    #task_room = int(timesteps.state.goal_room_idx[i])
        #    #task_object = int(timesteps.state.task_object_idx[i])
        #    #setting = 'single' if room_setting == 0 else 'multi'
        #    #category, color = maze_config['pairs'][task_room][task_object]
        #    #task_name = f'{setting} - {color} {category}'

        #    title = f'{task_name}\n'
        #    title += f't={i}\n'
        #    title += f'{actions_taken[i]}\n'
        #    title += f'r={timesteps.reward[i]}, $\\gamma={timesteps.discount[i]}$'
        #    return title

        #fig = plot_frames(
        #    timesteps=timesteps,
        #    frames=obs_images,
        #    panel_title_fn=panel_title_fn,
        #    ncols=6)
        #if wandb.run is not None:
        #    wandb.log(
        #        {f"learner_example/{key}/trajectory": wandb.Image(fig)})
        #plt.close(fig)

    def callback(d):
        n_updates = d.pop('n_updates')
        # [T, B] --> [T]
        if 'online' in d:
            d['online'] = jax.tree_map(lambda x: x[:, 0], d['online'])
            log_data(**d['online'], key='online')
        if 'dyna' in d:
            # [T, B, K, N] --> [K]
            # K = the simulation length
            # get entire simulation, starting at:
            #   T=0 (1st time-point)
            #   B=0 (1st batch sample)
            #   N=index(t_min) (simulation with lowest temperaturee)
            d['dyna'] = jax.tree_map(lambda x: x[0, 0, :, sim_idx], d['dyna'])
            log_data(**d['dyna'], key='dyna')

    # this will be the value after update is applied
    n_updates = data['n_updates'] + 1
    is_log_time = n_updates % config["LEARNER_EXTRA_LOG_PERIOD"] == 0

    jax.lax.cond(
        is_log_time,
        lambda d: jax.debug.callback(callback, d),
        lambda d: None,
        data)




class DynaAgentEnvModel(nn.Module):

    action_dim: int

    observation_encoder: nn.Module
    rnn: vbb.ScannedRNN
    env: environment.Environment
    env_params: environment.EnvParams
    num_q_layers: int = 1
    activation: str = 'relu'

    def setup(self):

        self.q_fn = MLP(
            hidden_dim=512,
            num_layers=self.num_q_layers + 1,
            out_dim=self.action_dim,
            activation=self.activation,
            activate_final=False,
            )

    def initialize(self, x: TimeStep):
        """Only used for initialization."""
        # [B, D]
        rng = jax.random.PRNGKey(0)
        batch_dims = (x.reward.shape[0],)
        rnn_state = self.initialize_carry(rng, batch_dims)
        predictions, rnn_state = self.__call__(rnn_state, x, rng)
        dummy_action = jnp.zeros(batch_dims, dtype=jnp.int32)
        self.apply_model(predictions.state, dummy_action, rng)

    def initialize_carry(self, *args, **kwargs):
        """Initializes the RNN state."""
        return self.rnn.initialize_carry(*args, **kwargs)

    def __call__(self, rnn_state, x: TimeStep, rng: jax.random.PRNGKey) -> Tuple[Predictions, RnnState]:

        embedding = self.observation_encoder(x.observation)

        rnn_in = vbb.RNNInput(obs=embedding, reset=x.first())
        rng, _rng = jax.random.split(rng)
        new_rnn_state, rnn_out = self.rnn(rnn_state, rnn_in, _rng)

        q_vals = self.q_fn(rnn_out)
        predictions = Predictions(
            q_vals=q_vals,
            state=AgentState(
                timestep=x,
                rnn_state=new_rnn_state)
            )

        return predictions, new_rnn_state

    def unroll(self, rnn_state, xs: TimeStep, rng: jax.random.PRNGKey) -> Tuple[Predictions, RnnState]:
        # rnn_state: [B]
        # xs: [T, B]

        embedding = jax.vmap(self.observation_encoder)(xs.observation)

        rnn_in = vbb.RNNInput(obs=embedding, reset=xs.first())
        rng, _rng = jax.random.split(rng)

        # [B, D], [T, B, D]
        new_rnn_state, new_rnn_states = self.rnn.unroll(rnn_state, rnn_in, _rng)

        rnn_out = self.rnn.output_from_state(new_rnn_states)
        q_vals = nn.BatchApply(self.q_fn)(rnn_out)
        predictions = Predictions(
            q_vals=q_vals,
            state=AgentState(
                timestep=xs,
                rnn_state=new_rnn_states)
            )
        return predictions, new_rnn_state

    def apply_model(
          self,
          state: AgentState,
          action: jnp.ndarray,
          rng: jax.random.PRNGKey,
      ) -> Tuple[Predictions, RnnState]:
        """This applies the model to each element in the state, action vectors.
        Args:
            state (State): states. [B, D]
            action (jnp.ndarray): actions to take on states. [B]
        Returns:
            Tuple[ModelOutput, State]: muzero outputs and new states for 
              each state state action pair.
        """
        # take one step forward in the environment
        B = action.shape[0]
        rng, rng_ = jax.random.split(rng)
        def env_step(s, a, rng_): 
           return self.env.step(rng_, s.timestep, a, self.env_params)
        next_timestep = jax.vmap(env_step)(
            state, action, jax.random.split(rng_, B))

        # compute predictions for next time-step
        rng, rng_ = jax.random.split(rng)
        predictions, new_rnn_state =  self.__call__(state.rnn_state, next_timestep, rng_)
        return predictions, new_rnn_state

    def unroll_model(
        self,
        state: AgentState,
        actions: jnp.ndarray,
        rng: jax.random.PRNGKey,
    ) -> Tuple[Predictions, RnnState]:
        """This applies the model recursively to the state using the sequence of actions.
        Args:
            state (State): states. [B, D]
            action (jnp.ndarray): actions to take on states. [T, B]
        Returns:
            Tuple[ModelOutput, State]: muzero outputs and new states for 
              each state state action pair.
        """

        def body_fn(model, carry, inputs):
            action = inputs
            state, rng_ = carry
            rng_, rng__ = jax.random.split(rng_)
            predictions, _ = model.apply_model(
                state, action, rng__)
            carry = (predictions.state, rng)
            return carry, predictions

        scan = nn.scan(
            body_fn,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0
        )
        return scan(self, (state, rng), actions)


def make_agent(
        config: dict,
        env: environment.Environment,
        env_params: environment.EnvParams,
        example_timestep: TimeStep,
        rng: jax.random.PRNGKey,
        model_env_params: environment.EnvParams,
        ObsEncoderCls: nn.Module = KeyroomObsEncoder,
        ) -> Tuple[nn.Module, Params, vbb.AgentResetFn]:

    model_env_params = model_env_params or env_params
    cell_type = config.get('RNN_CELL_TYPE', 'OptimizedLSTMCell')
    if cell_type.lower() == 'none':
        rnn = vbb.DummyRNN()
    else:
        rnn = vbb.ScannedRNN(
            hidden_dim=config.get("AGENT_RNN_DIM", 128),
            cell_type=cell_type,
            unroll_output_state=True,
            )
    agent = DynaAgentEnvModel(
        activation=config['ACTIVATION'],
        action_dim=env.num_actions(env_params),
        num_q_layers=config['NUM_Q_LAYERS'],
        observation_encoder=ObsEncoderCls(),
        rnn=rnn,
        env=env,
        env_params=env_params,
    )

    rng, _rng = jax.random.split(rng)
    network_params = agent.init(
        _rng, example_timestep,
        method=agent.initialize)

    def reset_fn(params, example_timestep, reset_rng):
      batch_dims = example_timestep.reward.shape
      return agent.apply(
          params,
          batch_dims=batch_dims,
          rng=reset_rng,
          method=agent.initialize_carry)

    return agent, network_params, reset_fn
