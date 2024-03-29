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


from library.wrappers import TimeStep
from singleagent import value_based_basics as vbb
from projects.humansf.networks import KeyroomObsEncoder


Agent = nn.Module
Params = flax.core.FrozenDict
AgentState = flax.struct.PyTreeNode
RNNInput = vbb.RNNInput
R2D2LossFn = qlearning.R2D2LossFn
Predictions = qlearning.Predictions
EpsilonGreedy = qlearning.EpsilonGreedy
make_optimizer = qlearning.make_optimizer
make_loss_fn_class = qlearning.make_loss_fn_class
make_actor = qlearning.make_actor

def extract_timestep_input(timestep: TimeStep):
  return RNNInput(
      obs=timestep.observation,
      done=timestep.last())


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
    init_scale: float
    cell_type: nn.RNNCellBase = nn.LSTMCell

    def setup(self):

        self.observation_encoder = KeyroomObsEncoder(self.hidden_dim)

        self.cell = self.cell_type(self.hidden_dim)

        self.rnn = vbb.ScannedRNN(
            hidden_dim=self.hidden_dim,
            cell=self.cell)

        self.q_fn = nn.Dense(self.action_dim, kernel_init=orthogonal(
            self.init_scale), bias_init=constant(0.0))

    def initialize(self, x: TimeStep):
        """Only used for initialization."""
        # [B, D]
        embedding = self.observation_encoder(x.observation)
        rnn_state = self.initialize_carry(embedding.shape)
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

        embedding = nn.BatchApply(self.observation_encoder)(x.obs)
        embedding = nn.relu(embedding)

        rnn_in = x._replace(obs=embedding)
        new_rnn_state, rnn_out = self.rnn.unroll(rnn_state, rnn_in)

        q_vals = nn.BatchApply(self.q_fn)(rnn_out)

        return Predictions(q_vals, rnn_out), new_rnn_state

    def initialize_carry(self, example_shape: Tuple[int]):
        return self.rnn.initialize_carry(example_shape)

def make_agent(
        config: dict,
        env: environment.Environment,
        env_params: environment.EnvParams,
        example_timestep: TimeStep,
        rng: jax.random.KeyArray) -> Tuple[Agent, Params, vbb.AgentResetFn]:

    agent = AgentRNN(
        action_dim=env.num_actions(env_params),
        hidden_dim=config["AGENT_HIDDEN_DIM"],
        init_scale=config['AGENT_INIT_SCALE'],
    )

    rng, _rng = jax.random.split(rng)
    network_params = agent.init(
        _rng, example_timestep, method=agent.initialize)

    def reset_fn(params, example_timestep):
      # always true
      batch_size = example_timestep.reward.shape
      example_shape = batch_size+(config["AGENT_HIDDEN_DIM"],)
      return agent.apply(
          params,
          example_shape,
          method=agent.initialize_carry)

    return agent, network_params, reset_fn
