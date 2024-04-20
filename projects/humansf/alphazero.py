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

from library import utils
from library import loggers

from projects.humansf.networks import KeyroomObsEncoder

from singleagent.basics import TimeStep
from singleagent import value_based_basics as vbb
from singleagent import alphazero as base_agent

Params = flax.core.FrozenDict
RnnState = jax.Array

make_optimizer = base_agent.make_optimizer
make_loss_fn_class = base_agent.make_loss_fn_class
make_actor = base_agent.make_actor
MLP = base_agent.MLP
Predictions = base_agent.Predictions
AgentState = base_agent.AgentState


#class AlphaZeroAgent(nn.Module):

#    action_dim: int
#    config: dict

#    rnn: vbb.ScannedRNN
#    env: environment.Environment
#    env_params: environment.EnvParams
#    test_env_params: environment.EnvParams

#    def setup(self):
#        self.

#        self.policy_fn = MLP(
#            hidden_dim=512, num_layers=1, out_dim=self.action_dim)
#        self.value_fn = MLP(
#            hidden_dim=512, num_layers=1, out_dim=self.config['NUM_BINS'])

#    def initialize(self, x: TimeStep):
#        """Only used for initialization."""
#        # [B, D]
#        rng = jax.random.PRNGKey(0)
#        batch_dims = (x.reward.shape[0],)
#        rnn_state = self.initialize_carry(rng, batch_dims)
#        predictions, rnn_state = self.__call__(rnn_state, x, rng)
#        dummy_action = jnp.zeros(batch_dims, dtype=jnp.uint32)
#        self.apply_model(predictions.state, dummy_action, rng)

#    def initialize_carry(self, *args, **kwargs):
#        """Initializes the RNN state."""
#        return self.rnn.initialize_carry(*args, **kwargs)

#    def __call__(self, rnn_state, x: TimeStep, rng: jax.random.KeyArray) -> Tuple[Predictions, RnnState]:

#        embedding = self.observation_encoder(x.observation)
#        embedding = nn.relu(embedding)

#        rnn_in = vbb.RNNInput(obs=embedding, reset=x.first())
#        rng, _rng = jax.random.split(rng)
#        new_rnn_state, rnn_out = self.rnn(rnn_state, rnn_in, _rng)

#        policy_logits = self.policy_fn(rnn_out)
#        value_logits = self.value_fn(rnn_out)
#        predictions = Predictions(
#            policy_logits=policy_logits,
#            value_logits=value_logits,
#            state=AgentState(
#                timestep=x,
#                rnn_state=new_rnn_state)
#            )

#        return predictions, new_rnn_state

#    def unroll(self, rnn_state, xs: TimeStep, rng: jax.random.KeyArray) -> Tuple[Predictions, RnnState]:
#        # rnn_state: [B]
#        # xs: [T, B]

#        embedding = jax.vmap(self.observation_encoder)(xs.observation)
#        embedding = nn.relu(embedding)

#        rnn_in = vbb.RNNInput(obs=embedding, reset=xs.first())
#        rng, _rng = jax.random.split(rng)
#        new_rnn_state, new_rnn_states = self.rnn.unroll(rnn_state, rnn_in, _rng)

#        rnn_out = new_rnn_states[1]
#        policy_logits = jax.vmap(self.policy_fn)(rnn_out)
#        value_logits = jax.vmap(self.value_fn)(rnn_out)
#        predictions = Predictions(
#            policy_logits=policy_logits,
#            value_logits=value_logits,
#            state=AgentState(
#                timestep=xs,
#                rnn_state=new_rnn_states)
#            )
#        return predictions, new_rnn_state

#    def apply_model(
#          self,
#          state: AgentState,
#          action: jnp.ndarray,
#          rng: jax.random.KeyArray,
#          evaluation: bool = False,
#      ) -> Tuple[Predictions, RnnState]:
#        """This applies the model to each element in the state, action vectors.

#        Args:
#            state (State): states. [1, D]
#            action (jnp.ndarray): actions to take on states. [1]

#        Returns:
#            Tuple[ModelOutput, State]: muzero outputs and new states for 
#              each state state action pair.
#        """
#        assert action.shape[0] == 1, 'function only accepts batchsize=1 due to inability to vmap over environment. please use vmap to get these dimensions.'
#        rng, rng_ = jax.random.split(rng)
#        env_params = self.test_env_params if evaluation else self.env_params
#        timestep = jax.tree_map(lambda x: x[0], state.timestep)
#        next_timestep = self.env.step(rng_, timestep, action[0], env_params)
#        next_timestep = jax.tree_map(lambda x: x[None], next_timestep)

#        rng, rng_ = jax.random.split(rng)
#        return self.__call__(state.rnn_state, next_timestep, rng_)


def make_logger(config: dict,
                env: environment.Environment,
                env_params: environment.EnvParams):

    def learner_log_extra(data: dict):
        def callback(d):
            if d['batch_index'] != 0:
                # called inside AlphaZeroLossFn:loss_fn
                # this function is called for every batch element.
                # only log first
                return

            rewards = d['data'].timestep.reward
            values = d['values']
            values_target = d['values_targets']

            # Create a figure with three subplots
            nplots = 4
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(
                nplots, 1, figsize=(5, 3*nplots))

            # Plot rewards and q-values in the top subplot
            def format(ax):
                ax.set_xlabel('Time')
                ax.grid(True)
                ax.set_xticks(range(0, len(rewards), 1))

            # Plot rewards and q-values in the top subplot
            ax1.plot(rewards, label='Rewards')
            ax1.plot(values, label='Value Predictions')
            ax1.plot(values_target, label='Value Targets')
            format(ax1)
            ax1.set_title('Rewards and Values')
            ax1.legend()

            # Plot TD errors in the middle subplot
            ax2.plot(d['td_errors'])
            format(ax2)
            ax2.set_title('TD Errors')

            # Plot Value-loss in the bottom subplot
            ax3.plot(d['value_loss'])
            format(ax3)
            ax3.set_title('Value Loss')

            # Plot Value-loss in the bottom subplot
            ax4.plot(d['policy_loss'])
            format(ax4)
            ax4.set_title('Policy Loss')

            # Adjust the spacing between subplots
            plt.tight_layout()
            # log
            if wandb.run is not None:
                wandb.log({f"learner_details/losses": wandb.Image(fig)})
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
        learner_log_extra=learner_log_extra,
    )


def make_agent(
        config: dict,
        env: environment.Environment,
        env_params: environment.EnvParams,
        example_timestep: TimeStep,
        rng: jax.random.KeyArray,
        test_env_params: environment.EnvParams,
        ) -> Tuple[nn.Module, Params, vbb.AgentResetFn]:

    agent = base_agent.AlphaZeroAgent(
        action_dim=env.num_actions(env_params),
        observation_encoder=KeyroomObsEncoder(
            hidden_dim=config["AGENT_HIDDEN_DIM"],
            init=config.get('ENCODER_INIT', 'word_init'),
            image_hidden_dim=config.get('IMAGE_HIDDEN', 512),
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

