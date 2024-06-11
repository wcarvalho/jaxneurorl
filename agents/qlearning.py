"""
Recurrent Q-learning.
"""



import functools
import os
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import NamedTuple, Dict, Union, Optional, Tuple, Callable

import chex

import optax
import flax.linen as nn
import wandb
import matplotlib.pyplot as plt


import flax
from flax import struct
import rlax
from gymnax.environments import environment

from library import losses
from library import loggers
from agents.basics import TimeStep
from agents import value_based_basics as vbb

def extract_timestep_input(timestep: TimeStep):
  return RNNInput(
      obs=timestep.observation,
      reset=timestep.first())

Agent = nn.Module
Params = flax.core.FrozenDict
AgentState = flax.struct.PyTreeNode
RNNInput = vbb.RNNInput

@struct.dataclass
class R2D2LossFn(vbb.RecurrentLossFn):

  """Loss function of R2D2.
  
  https://openreview.net/forum?id=r1lyTjAqYX
  """

  tx_pair: rlax.TxPair = rlax.IDENTITY_PAIR
  extract_q: Callable[[jax.Array], jax.Array] = lambda preds: preds.q_vals


  def error(self, data, online_preds, online_state, target_preds, target_state, steps, **kwargs):
    """R2D2 learning.
    """

    float = lambda x: x.astype(jnp.float32)
    # Get value-selector actions from online Q-values for double Q-learning.
    selector_actions = jnp.argmax(self.extract_q(online_preds), axis=-1)  # [T+1, B]

    # Preprocess discounts & rewards.
    discounts = float(data.discount)*self.discount
    lambda_ = jnp.ones_like(data.discount)*self.lambda_
    rewards = float(data.reward)
    is_last = float(data.is_last)

    # Get N-step transformed TD error and loss.
    batch_td_error_fn = jax.vmap(
        losses.q_learning_lambda_td,
        in_axes=1,
        out_axes=1)

    # [T, B]
    q_t, target_q_t = batch_td_error_fn(
        self.extract_q(online_preds)[:-1],  # [T+1] --> [T]
        data.action[:-1],    # [T+1] --> [T]
        self.extract_q(target_preds)[1:],  # [T+1] --> [T]
        selector_actions[1:],  # [T+1] --> [T]
        rewards[1:],        # [T+1] --> [T]
        discounts[1:],
        is_last[1:],
        lambda_[1:])      # [T+1] --> [T]

    # ensure target = 0 when episode terminates
    target_q_t = target_q_t*data.discount[:-1]
    batch_td_error = target_q_t - q_t

    # ensure loss = 0 when episode truncates
    # truncated if FINAL time-step but data.discount = 1.0, something like [1,1,2,1,1]
    truncated = (data.discount+is_last) > 1  # truncated is discount on AND is last
    loss_mask = (1-truncated).astype(batch_td_error.dtype)[:-1]
    batch_td_error = batch_td_error*loss_mask

    # [T, B]
    batch_loss = 0.5 * jnp.square(batch_td_error)

    # [B]
    batch_loss_mean = (batch_loss*loss_mask).mean(0)

    metrics = {
        '0.q_loss': batch_loss.mean(),
        '0.q_td': jnp.abs(batch_td_error).mean(),
        '1.reward': rewards[1:].mean(),
        'z.q_mean': self.extract_q(online_preds).mean(),
        'z.q_var': self.extract_q(online_preds).var(),
        }

    if self.logger.learner_log_extra is not None:
        self.logger.learner_log_extra({
        'data': data,
        'td_errors': batch_td_error,                 # [T]
        'mask': loss_mask,                 # [T]
        'q_values': self.extract_q(online_preds),    # [T, B]
        'q_loss': batch_loss,                        #[ T, B]
        'q_target': target_q_t,
        'n_updates': steps,
        })

    return batch_td_error, batch_loss_mean, metrics  # [T-1, B], [B]


def make_logger(config: dict,
                env: environment.Environment,
                env_params: environment.EnvParams):

    def qlearner_logger(data: dict):
        def callback(d):
            if wandb.run is None: return
            n_updates = d.pop('n_updates')

            # Extract the relevant data
            # only use data from batch dim = 0
            # [T, B, ...] --> # [T, ...]
            d_ = jax.tree_map(lambda x: x[:, 0], d)

            rewards = d_['data'].timestep.reward
            actions = d_['data'].action
            q_values = d_['q_values']
            q_target = d_['q_target']
            q_values_taken = rlax.batched_index(q_values, actions)

            # Create a figure with three subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))

            def format(ax):
                ax.set_xlabel('Time')
                ax.grid(True)
                ax.set_xticks(range(0, len(rewards), 1))

            # Plot rewards and q-values in the top subplot
            ax1.plot(rewards, label='Rewards')
            ax1.plot(q_values_taken, label='Q-Values')
            ax1.plot(q_target, label='Q-Targets')
            format(ax1)
            ax1.set_title('Rewards and Q-Values')
            ax1.legend()

            # Plot TD errors in the middle subplot
            ax2.plot(d_['td_errors'])
            format(ax2)
            ax2.set_title('TD Errors')

            # Plot Q-loss in the bottom subplot
            ax3.plot(d_['q_loss'])
            format(ax3)
            ax3.set_title('Q-Loss')

            # Adjust the spacing between subplots
            plt.tight_layout()
            # log
            wandb.log({f"learner_details/q-values": wandb.Image(fig)})
            plt.close(fig)


        # this will be the value after update is applied
        n_updates = data['n_updates'] + 1
        is_log_time = n_updates % config["LEARNER_EXTRA_LOG_PERIOD"] == 0

        jax.lax.cond(
            is_log_time,
            lambda d: jax.debug.callback(callback, d),
            lambda d: None,
            data)


    return loggers.Logger(
        gradient_logger=loggers.default_gradient_logger,
        learner_logger=loggers.default_learner_logger,
        experience_logger=loggers.default_experience_logger,
        learner_log_extra=qlearner_logger,
    )

class Predictions(NamedTuple):
    q_vals: jax.Array
    rnn_states: jax.Array

class Block(nn.Module):
  features: int

  @nn.compact
  def __call__(self, x, _):
    x = nn.Dense(self.features, use_bias=False)(x)
    x = jax.nn.relu(x)
    return x, None

class MLP(nn.Module):
  hidden_dim: int
  out_dim: Optional[int] = None
  num_layers: int = 1

  @nn.compact
  def __call__(self, x):
    for _ in range(self.num_layers):
        x, _ = Block(self.hidden_dim)(x, None)

    x = nn.Dense(self.out_dim or self.hidden_dim, use_bias=False)(x)
    return x

class RnnAgent(nn.Module):
    action_dim: int
    hidden_dim: int
    init_scale: float
    rnn: vbb.ScannedRNN

    def setup(self):
        self.observation_encoder = MLP(
           hidden_dim=self.hidden_dim, num_layers=1)
        self.q_fn = MLP(hidden_dim=512, num_layers=1, out_dim=self.action_dim)

    def initialize(self, x: TimeStep):
        """Only used for initialization."""
        # [B, D]
        rng = jax.random.PRNGKey(0)
        batch_dims = (x.reward.shape[0],)
        rnn_state = self.initialize_carry(rng, batch_dims)

        return self.__call__(rnn_state, x, rng)

    def initialize_carry(self, *args, **kwargs):
        """Initializes the RNN state."""
        return self.rnn.initialize_carry(*args, **kwargs)

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

        embedding = self.observation_encoder(xs.obs)
        embedding = nn.relu(embedding)

        rnn_in = xs._replace(obs=embedding)
        rng, _rng = jax.random.split(rng)
        new_rnn_state, rnn_out = self.rnn.unroll(rnn_state, rnn_in, _rng)

        q_vals = self.q_fn(rnn_out)

        return Predictions(q_vals, rnn_out), new_rnn_state

def epsilon_greedy_act(q, eps, key):
    key_a, key_e   = jax.random.split(key, 2) # a key for sampling random actions and one for picking
    greedy_actions = jnp.argmax(q, axis=-1) # get the greedy actions 
    random_actions = jax.random.randint(key_a, shape=greedy_actions.shape, minval=0, maxval=q.shape[-1]) # sample random actions
    pick_random    = jax.random.uniform(key_e, greedy_actions.shape)<eps # pick which actions should be random
    chosen_actions = jnp.where(pick_random, random_actions, greedy_actions)
    return chosen_actions

class LinearDecayEpsilonGreedy:
    """Epsilon Greedy action selection"""

    def __init__(self, start_e: float, end_e: float, duration: int):
        self.start_e  = start_e
        self.end_e    = end_e
        self.duration = duration
        self.slope    = (end_e - start_e) / duration

    @partial(jax.jit, static_argnums=0)
    def get_epsilon(self, t: int):
        e = self.slope*t + self.start_e
        return jnp.clip(e, self.end_e)

    @partial(jax.jit, static_argnums=0)
    def choose_actions(self, q_vals: jnp.ndarray, t: int, rng: chex.PRNGKey):

        eps = self.get_epsilon(t)
        rng = jax.random.split(rng, q_vals.shape[0])
        return jax.vmap(epsilon_greedy_act, in_axes=(0, None, 0))(q_vals, eps, rng)

class FixedEpsilonGreedy:
    """Epsilon Greedy action selection"""

    def __init__(self, epsilons: float):
        self.epsilons = epsilons

    @partial(jax.jit, static_argnums=0)
    def choose_actions(self, q_vals: jnp.ndarray, t: int, rng: chex.PRNGKey):

        rng = jax.random.split(rng, q_vals.shape[0])
        return jax.vmap(epsilon_greedy_act, in_axes=(0, 0, 0))(q_vals, self.epsilons, rng)

def make_rnn_agent(
        config: dict,
        env: environment.Environment,
        env_params: environment.EnvParams,
        example_timestep: TimeStep,
        rng: jax.random.KeyArray) -> Tuple[nn.Module, Params, vbb.AgentResetFn]:

    agent = RnnAgent(
        action_dim=env.action_space(env_params).n,
        hidden_dim=config["AGENT_HIDDEN_DIM"],
        init_scale=config['AGENT_INIT_SCALE'],
        rnn=vbb.ScannedRNN(hidden_dim=config["AGENT_HIDDEN_DIM"])
    )

    rng, _rng = jax.random.split(rng)
    network_params = agent.init(
        _rng, example_timestep,
        method=agent.initialize)

    def reset_fn(params, example_timestep, reset_rng):
      batch_dims = (example_timestep.reward.shape[0],)
      return agent.apply(
          params,
          batch_dims=batch_dims,
          rng=reset_rng,
          method=agent.initialize_carry)

    return agent, network_params, reset_fn

def make_mlp_agent(
        config: dict,
        env: environment.Environment,
        env_params: environment.EnvParams,
        example_timestep: TimeStep,
        rng: jax.random.KeyArray) -> Tuple[nn.Module, Params, vbb.AgentResetFn]:

    agent = RnnAgent(
        action_dim=env.action_space(env_params).n,
        hidden_dim=config["AGENT_HIDDEN_DIM"],
        init_scale=config['AGENT_INIT_SCALE'],
        rnn=vbb.DummyRNN()
    )

    rng, _rng = jax.random.split(rng)
    network_params = agent.init(
        _rng, example_timestep,
        method=agent.initialize)

    def reset_fn(params, example_timestep, reset_rng):
      del params
      del reset_rng
      batch_dims = example_timestep.observation.shape[:-1]
      return jnp.zeros(batch_dims)

    return agent, network_params, reset_fn

def make_optimizer(config: dict) -> optax.GradientTransformation:
  def linear_schedule(count):
      frac = 1.0 - (count / config["NUM_UPDATES"])
      return config["LR"] * frac

  lr = linear_schedule if config.get("LR_LINEAR_DECAY", False) else config["LR"]

  return optax.chain(
      optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
      optax.adam(learning_rate=lr, eps=config['EPS_ADAM'])
  )

def make_loss_fn_class(config) -> vbb.RecurrentLossFn:
  return functools.partial(
     R2D2LossFn,
     discount=config['GAMMA'])

def make_actor(config: dict, agent: Agent, rng: jax.random.KeyArray) -> vbb.Actor:
    fixed_epsilon = config.get('FIXED_EPSILON', 1)
    assert fixed_epsilon in (0, 1, 2)
    if fixed_epsilon:
        ## BELOW was copied from ACME
        if fixed_epsilon == 1: 
            vals = np.logspace(
                    start=config.get('EPSILON_MIN', 1),
                    stop=config.get('EPSILON_MAX', 3),
                    num=config.get('NUM_EPSILONS', 256),
                    base=config.get('EPSILON_BASE', .1))
        else:
            # BELOW is in range of ~(.9,.1) 
            vals = np.logspace(
                    num=config.get('NUM_EPSILONS', 256),
                    start=config.get('EPSILON_MIN', .05),
                    stop=config.get('EPSILON_MAX', .9),
                    base=config.get('EPSILON_BASE', .1))
        epsilons = jax.random.choice(
            rng, vals, shape=(config['NUM_ENVS'],))

        explorer = FixedEpsilonGreedy(epsilons)
    else:
        explorer = LinearDecayEpsilonGreedy(
            start_e=config["EPSILON_START"],
            end_e=config["EPSILON_FINISH"],
            duration=config["EPSILON_ANNEAL_TIME"]
        )

    def actor_step(
        train_state: vbb.TrainState,
        agent_state: jax.Array,
        timestep: TimeStep,
        rng: jax.random.KeyArray):
        preds, agent_state = agent.apply(
            train_state.params, agent_state, timestep, rng)

        action = explorer.choose_actions(
            preds.q_vals, train_state.timesteps, rng)

        return preds, action, agent_state

    def eval_step(
        train_state: vbb.TrainState,
        agent_state: jax.Array,
        timestep: TimeStep,
        rng: jax.random.KeyArray):
        preds, agent_state = agent.apply(
            train_state.params, agent_state, timestep, rng)

        action = preds.q_vals.argmax(-1)

        return preds, action, agent_state

    return vbb.Actor(train_step=actor_step, eval_step=eval_step)

make_train_preloaded = functools.partial(
   vbb.make_train,
   make_agent=make_rnn_agent,
   make_optimizer=make_optimizer,
   make_loss_fn_class=make_loss_fn_class,
   make_actor=make_actor,
   make_logger=make_logger,
)