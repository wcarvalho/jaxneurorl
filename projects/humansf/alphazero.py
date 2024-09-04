"""
Alpha-Zero
"""
import functools
from typing import Callable, Dict, List, Optional, Tuple, Union

import distrax
import flax
from flax import struct
import flax.linen as nn
from gymnax.environments import environment

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mctx
import optax
import rlax
import wandb

from jaxneurorl import utils
from jaxneurorl import loggers

from projects.humansf.networks import KeyroomObsEncoder

from jaxneurorl.agents.basics import TimeStep
from jaxneurorl.agents import value_based_basics as vbb
from jaxneurorl.agents import alphazero as base_agent

Params = flax.core.FrozenDict
RnnState = jax.Array

make_optimizer = base_agent.make_optimizer
make_loss_fn_class = base_agent.make_loss_fn_class
make_actor = base_agent.make_actor
MLP = base_agent.MLP
Predictions = base_agent.Predictions
AgentState = base_agent.AgentState


def make_agent(
        config: dict,
        env: environment.Environment,
        env_params: environment.EnvParams,
        example_timestep: TimeStep,
        rng: jax.random.PRNGKey,
        test_env_params: environment.EnvParams,
        ObsEncoderCls: nn.Module = KeyroomObsEncoder,
        ) -> Tuple[nn.Module, Params, vbb.AgentResetFn]:

    agent = base_agent.AlphaZeroAgent(
        action_dim=env.num_actions(env_params),
        observation_encoder=ObsEncoderCls(
            embed_hidden_dim=config["AGENT_HIDDEN_DIM"],
            init=config.get('ENCODER_INIT', 'word_init'),
            grid_hidden_dim=config.get('GRID_HIDDEN', 256),
            num_embed_layers=config['NUM_EMBED_LAYERS'],
            num_grid_layers=config['NUM_GRID_LAYERS'],
            num_joint_layers=config['NUM_ENCODER_LAYERS'],
            include_extras=config.get('ENC_INCLUDE_EXTRAS', False),
        ),
        rnn=vbb.ScannedRNN(
            hidden_dim=config.get("AGENT_RNN_DIM", 128),
            unroll_output_state=True,
            ),
        num_bins=config['NUM_BINS'],
        env=env,
        env_params=env_params,
        test_env_params=test_env_params,
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

