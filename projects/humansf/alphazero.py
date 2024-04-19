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


class AlphaZeroAgent(nn.Module):

    action_dim: int
    config: dict

    rnn: vbb.ScannedRNN
    env: environment.Environment
    env_params: environment.EnvParams
    test_env_params: environment.EnvParams

    num_bins: int = 101

    def setup(self):
        self.observation_encoder = KeyroomObsEncoder(
            hidden_dim=self.hidden_dim,
            init=self.config.get('ENCODER_INIT', 'word_init'),
            image_hidden_dim=self.config.get('IMAGE_HIDDEN', 512),
        )

        self.policy_fn = MLP(hidden_dim=512, num_layers=1, out_dim=self.action_dim)
        self.value_fn = MLP(hidden_dim=512, num_layers=1, out_dim=self.num_bins)

    def initialize(self, x: TimeStep):
        """Only used for initialization."""
        # [B, D]
        rng = jax.random.PRNGKey(0)
        batch_dims = x.observation.shape[:-1]
        rnn_state = self.initialize_carry(rng, batch_dims)
        predictions, rnn_state = self.__call__(rnn_state, x, rng)
        dummy_action = jnp.zeros(batch_dims, dtype=jnp.uint32)
        self.apply_model(predictions.state, dummy_action, rng)

    def initialize_carry(self, *args, **kwargs):
        """Initializes the RNN state."""
        return self.rnn.initialize_carry(*args, **kwargs)

    def __call__(self, rnn_state, x: TimeStep, rng: jax.random.KeyArray) -> Tuple[Predictions, RnnState]:

        embedding = self.observation_encoder(x.observation)
        embedding = nn.relu(embedding)

        rnn_in = vbb.RNNInput(obs=embedding, reset=x.first())
        rng, _rng = jax.random.split(rng)
        new_rnn_state, rnn_out = self.rnn(rnn_state, rnn_in, _rng)

        policy_logits = self.policy_fn(rnn_out)
        value_logits = self.value_fn(rnn_out)
        predictions = Predictions(
            policy_logits=policy_logits,
            value_logits=value_logits,
            state=AgentState(
                timestep=x,
                rnn_state=rnn_out)
            )

        return predictions, new_rnn_state

    def unroll(self, rnn_state, xs: TimeStep, rng: jax.random.KeyArray) -> Tuple[Predictions, RnnState]:
        # rnn_state: [B]
        # xs: [T, B]

        embedding = jax.vmap(self.observation_encoder)(xs.observation)
        embedding = nn.relu(embedding)

        rnn_in = vbb.RNNInput(obs=embedding, reset=xs.first())
        rng, _rng = jax.random.split(rng)
        new_rnn_state, rnn_out = self.rnn.unroll(rnn_state, rnn_in, _rng)

        policy_logits = jax.vmap(self.policy_fn)(rnn_out)
        value_logits = jax.vmap(self.value_fn)(rnn_out)
        predictions = Predictions(
            policy_logits=policy_logits,
            value_logits=value_logits,
            state=AgentState(
                timestep=xs,
                rnn_state=rnn_out)
            )
        return predictions, new_rnn_state

    def apply_model(
          self,
          state: AgentState,  # [B, D]
          action: jnp.ndarray,  # [B]
          rng: jax.random.KeyArray,
          evaluation: bool = False,
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
        env_params = self.test_env_params if evaluation else self.env_params
        env_step = lambda s, a, rng_: self.env.step(rng_, s.timestep, a, env_params)
        next_timestep = jax.vmap(env_step)(state, action, jax.random.split(rng_, B))

        # compute value and policy for the next time-step
        rng, rng_ = jax.random.split(rng)
        return self.__call__(state.rnn_state, next_timestep, rng_)


def make_agent(
        config: dict,
        env: environment.Environment,
        env_params: environment.EnvParams,
        example_timestep: TimeStep,
        rng: jax.random.KeyArray,
        test_env_params: Optional[environment.EnvParams] = None,
        ) -> Tuple[nn.Module, Params, vbb.AgentResetFn]:

    test_env_params = test_env_params or env_params
    agent = AlphaZeroAgent(
        action_dim=env.action_space(env_params).n,
        config=config,
        rnn=vbb.ScannedRNN(hidden_dim=config.get("AGENT_RNN_DIM", 128)),
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
      batch_dims = example_timestep.observation.shape[:-1]
      return agent.apply(
          params,
          batch_dims=batch_dims,
          rng=reset_rng,
          method=agent.initialize_carry)

    return agent, network_params, reset_fn

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

def make_train_preloaded(config, test_env_params=None):
    max_value = config.get('MAX_VALUE', 10)
    num_bins = config['NUM_BINS']

    discretizer = utils.Discretizer(
        max_value=max_value,
        num_bins=num_bins,
        min_value=-max_value)

    mcts_policy = functools.partial(
        mctx.gumbel_muzero_policy,
        max_depth=config.get('MAX_SIM_DEPTH', None),
        num_simulations=config.get('NUM_SIMULATIONS', 4),
        gumbel_scale=config.get('GUMBEL_SCALE', 1.0))

    return functools.partial(
        vbb.make_train,
        make_agent=functools.partial(
            make_agent,
            test_env_params=test_env_params),
        make_optimizer=make_optimizer,
        make_loss_fn_class=functools.partial(
            make_loss_fn_class,
            discretizer=discretizer),
        make_actor=functools.partial(
            make_actor,
            discretizer=discretizer,
            mcts_policy=mcts_policy),
        make_logger=make_logger,
    )
