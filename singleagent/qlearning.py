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
import dataclasses
from ray import tune

import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import flashbax as fbx
import wandb
import hydra

import flax
import rlax
from omegaconf import OmegaConf
from safetensors.flax import save_file
from flax.traverse_util import flatten_dict
from gymnax.environments import environment


from library.wrappers import TimeStep
from singleagent import value_based_basics as vbb

def extract_timestep_input(timestep: TimeStep):
  return RNNInput(
      obs=timestep.observation,
      done=timestep.last())

Agent = nn.Module
Params = flax.core.FrozenDict
AgentState = flax.struct.PyTreeNode
RNNInput = vbb.RNNInput

@dataclasses.dataclass
class R2D2LossFn(vbb.RecurrentLossFn):

  """Loss function of R2D2.
  
  https://openreview.net/forum?id=r1lyTjAqYX
  """

  tx_pair: rlax.TxPair = rlax.IDENTITY_PAIR
  extract_q: Callable[[jax.Array], jax.Array] = lambda preds: preds.q_vals
  bootstrap_n: int = 5

  def error(self, data, online_preds, online_state, target_preds, target_state, **kwargs):
    """R2D2 learning.
    """
    # Get value-selector actions from online Q-values for double Q-learning.
    selector_actions = jnp.argmax(self.extract_q(online_preds), axis=-1)  # [T+1, B]
    # Preprocess discounts & rewards.
    discounts = (data.discount * self.discount).astype(self.extract_q(online_preds).dtype)
    rewards = data.reward
    rewards = rewards.astype(self.extract_q(online_preds).dtype)

    # Get N-step transformed TD error and loss.
    batch_td_error_fn = jax.vmap(
        functools.partial(
            rlax.transformed_n_step_q_learning,
            n=self.bootstrap_n,
            tx_pair=self.tx_pair),
        in_axes=1,
        out_axes=1)

    batch_td_error = batch_td_error_fn(
        self.extract_q(online_preds)[:-1],  # [T+1] --> [T]
        data.action[:-1],    # [T+1] --> [T]
        self.extract_q(target_preds)[1:],  # [T+1] --> [T]
        selector_actions[1:],  # [T+1] --> [T]
        rewards[1:],        # [T+1] --> [T]
        discounts[1:])      # [T+1] --> [T]

    # NOTE: discount AT terminal state = 0. discount BEFORE terminal state = 1.
    # AT terminal state, don't want loss from terminal to next because that crosses
    # episode boundaries. so use discount[:-1] for making mask.
    mask = data.discount[:-1]  # if 0, episode ending
    batch_loss = 0.5 * jnp.square(batch_td_error).mean(axis=0)

    batch_loss = vbb.maked_mean(batch_loss, mask)

    metrics = {
        '0.q_loss': batch_loss.mean(),
        '0.q_td': jnp.abs(batch_td_error).mean(),
        '1.reward': rewards.mean(),
        'z.q_mean': self.extract_q(online_preds).mean(),
        'z.q_var': self.extract_q(online_preds).var(),
        }

    return batch_td_error, batch_loss, metrics  # [T-1, B], [B]

class Predictions(NamedTuple):
    q_vals: jax.Array
    rnn_states: jax.Array

class AgentRNN(nn.Module):
    # homogenous agent for parameters sharing, assumes all agents have same obs and action dim
    action_dim: int
    hidden_dim: int
    init_scale: float
    cell_type: nn.RNNCellBase = nn.LSTMCell

    def setup(self):
        self.observation_encoder = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))
        self.cell = self.cell_type(self.hidden_dim)

        self.rnn = vbb.ScannedRNN(
            hidden_dim=self.hidden_dim,
            cell=self.cell)

        self.q_fn = nn.Dense(self.action_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))

    def initialize(self, x: TimeStep):
        """Only used for initialization."""
        # [B, D]
        rnn_state = self.initialize_carry(x.observation.shape)
        return self.__call__(rnn_state, x)

    def __call__(self, rnn_state, x: TimeStep):
        x = extract_timestep_input(x)

        embedding = self.observation_encoder(x.obs)
        embedding = nn.relu(embedding)

        rnn_in = x._replace(obs=embedding)
        rnn_out, new_rnn_state = self.rnn(rnn_state, rnn_in)

        q_vals = self.q_fn(rnn_out)

        return Predictions(q_vals, rnn_out), new_rnn_state

    def unroll(self, rnn_state, x: TimeStep):
        # rnn_state: [B]
        # x: [T, B]
        x = extract_timestep_input(x)

        embedding = self.observation_encoder(x.obs)
        embedding = nn.relu(embedding)

        rnn_in = x._replace(obs=embedding)
        new_rnn_state, rnn_out = self.rnn.unroll(rnn_state, rnn_in)

        q_vals = self.q_fn(rnn_out)

        return Predictions(q_vals, rnn_out), new_rnn_state

    def initialize_carry(self, example_shape: Tuple[int]):
        return self.rnn.initialize_carry(example_shape)

class EpsilonGreedy:
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

        def explore(q, eps, key):
            key_a, key_e   = jax.random.split(key, 2) # a key for sampling random actions and one for picking
            greedy_actions = jnp.argmax(q, axis=-1) # get the greedy actions 
            random_actions = jax.random.randint(key_a, shape=greedy_actions.shape, minval=0, maxval=q.shape[-1]) # sample random actions
            pick_random    = jax.random.uniform(key_e, greedy_actions.shape)<eps # pick which actions should be random
            chosed_actions = jnp.where(pick_random, random_actions, greedy_actions)
            return chosed_actions

        eps = self.get_epsilon(t)
        rng = jax.random.split(rng, q_vals.shape[0])
        return jax.vmap(explore, in_axes=(0, None, 0))(q_vals, eps, rng)

def make_agent(
        config: dict,
        env: environment.Environment,
        env_params: environment.EnvParams,
        example_timestep: TimeStep,
        rng: jax.random.KeyArray) -> Tuple[nn.Module, Params, vbb.ResetFn]:

    agent = AgentRNN(
        action_dim=env.action_space(env_params).n,
        hidden_dim=config["AGENT_HIDDEN_DIM"],
        init_scale=config['AGENT_INIT_SCALE'],
    )

    rng, _rng = jax.random.split(rng)
    network_params = agent.init(
        _rng, example_timestep, method=agent.initialize)

    def reset_fn(params, example_timestep):
      return agent.apply(
          params,
          example_timestep.observation.shape,
          method=agent.initialize_carry)

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

def make_actor(config: dict, agent: Agent) -> vbb.Actor:
  explorer = EpsilonGreedy(
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
        train_state.params, agent_state, timestep)

    action = explorer.choose_actions(
        preds.q_vals, train_state.timesteps, rng)

    return preds, action, agent_state

  def eval_step(
        train_state: vbb.TrainState,
        agent_state: jax.Array,
        timestep: TimeStep,
        rng: jax.random.KeyArray):
    del rng
    preds, agent_state = agent.apply(
        train_state.params, agent_state, timestep)

    action = preds.q_vals.argmax(-1)

    return preds, action, agent_state

  return vbb.Actor(actor_step=actor_step, eval_step=eval_step)

make_train_preloaded = functools.partial(
   vbb.make_train,
   make_agent=make_agent,
   make_optimizer=make_optimizer,
   make_loss_fn_class=make_loss_fn_class,
   make_actor=make_actor
)