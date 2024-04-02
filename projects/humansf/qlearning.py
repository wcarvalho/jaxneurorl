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


from singleagent.basics import TimeStep
from singleagent import value_based_basics as vbb
from projects.humansf.networks import KeyroomObsEncoder


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

class Block(nn.Module):
  features: int

  @nn.compact
  def __call__(self, x, _):
    x = nn.Dense(self.features)(x)
    x = jax.nn.relu(x)
    return x, None

class MLP(nn.Module):
  hidden_dim: int
  out_dim: int
  num_layers: int = 1

  @nn.compact
  def __call__(self, x):
    ScanBlock = nn.scan(
      Block, variable_axes={'params': 0}, split_rngs={'params': True},
      length=self.num_layers)

    y, _ = ScanBlock(self.hidden_dim)(x, None)
    y = nn.Dense(self.out_dim)(y)
    return y

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

        self.q_fn = MLP(
           hidden_dim=self.hidden_dim,
           num_layers=2,
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
