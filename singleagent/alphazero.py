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
import mctx

from library import utils

from singleagent.basics import TimeStep
from singleagent import value_based_basics as vbb

Params = flax.core.FrozenDict
RnnState = jax.Array


@struct.dataclass
class AgentState:
    timestep: jax.Array
    rnn_state: jax.Array

@struct.dataclass
class Predictions:
    policy_logits: jax.Array
    value_logits: jax.Array
    state: AgentState


def model_step(params: Params,
               rng_key: jax.Array,
               action: jax.Array,
               state: jax.Array,
               discount: jax.Array,
               agent: nn.Module,
               discretizer: utils.Discretizer,
               evaluation: bool = False):
  """One simulation step in MCTS."""
  rng_key, model_key = jax.random.split(rng_key)
  predictions, _ = agent.apply(
      params, state=state, action=action,
      rng=model_key, evaluation=evaluation,
      method=agent.apply_model,
  )

  recurrent_fn_output = mctx.RecurrentFnOutput(
      reward=predictions.state.timestep.reward,
      discount=discount,
      prior_logits=predictions.policy_logits,
      value=discretizer.logits_to_scalar(predictions.value_logits),
  )
  return recurrent_fn_output, predictions.state

@struct.dataclass
class AlphaZeroLossFn(vbb.RecurrentLossFn):
    """Computes AlphaZero loss. 
    """
    discretizer: utils.Discretizer = utils.Discretizer(
      max_value=10, num_bins=101)

    #mcts_policy: Union[
    #  mctx.muzero_policy,
    #  mctx.gumbel_muzero_policy] = mctx.gumbel_muzero_policy

    #simulation_steps : int = 5  # how many time-steps of simulation to learn model with
    ##reanalyze_ratio : float = .5  # how often to learn from MCTS data vs. experience
    #value_target_source: str = 'return'

    #root_policy_coef: float = 1.0
    #root_value_coef: float = 0.25

    #state_from_preds: Callable[
    #  [Predictions], jax.Array] = lambda preds: preds.state

    def error(self, data, online_preds, online_state, target_preds, target_state, steps, **kwargs):
        import ipdb; ipdb.set_trace()


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

def extract_timestep_input(timestep: TimeStep):
  return vbb.RNNInput(
      obs=timestep.observation,
      reset=timestep.first())

class AlphaZeroAgent(nn.Module):

    action_dim: int
    hidden_dim: int
    init_scale: float

    rnn: vbb.ScannedRNN

    env: environment.Environment
    env_params: environment.EnvParams
    test_env_params: environment.EnvParams

    num_bins: int = 101

    def setup(self):
        self.observation_encoder = MLP(
           hidden_dim=self.hidden_dim, num_layers=1)

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

        embedding = nn.BatchApply(self.observation_encoder)(xs.observation)
        embedding = nn.relu(embedding)

        rnn_in = vbb.RNNInput(obs=embedding, reset=xs.first())
        rng, _rng = jax.random.split(rng)
        new_rnn_state, rnn_out = self.rnn.unroll(rnn_state, rnn_in, _rng)

        policy_logits = nn.BatchApply(self.policy_fn)(rnn_out)
        value_logits = nn.BatchApply(self.value_fn)(rnn_out)
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
        hidden_dim=config["AGENT_HIDDEN_DIM"],
        init_scale=config['AGENT_INIT_SCALE'],
        rnn=vbb.ScannedRNN(hidden_dim=config["AGENT_HIDDEN_DIM"]),
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


def make_loss_fn_class(
        config,
        discretizer: utils.Discretizer) -> vbb.RecurrentLossFn:
  return functools.partial(
     AlphaZeroLossFn,
     discretizer=discretizer,
     discount=config['GAMMA'])


def make_actor(
      config: dict,
      agent: nn.Module,
      rng: jax.random.KeyArray,
      discretizer: utils.Discretizer,
      mcts_policy: mctx.gumbel_muzero_policy,
      ) -> vbb.Actor:
    del config
    del rng

    def actor_step(
            train_state: vbb.TrainState,
            agent_state: jax.Array,
            timestep: TimeStep,
            rng: jax.random.KeyArray,
            evaluation: bool = False,
            ):
        preds, agent_state = agent.apply(
            train_state.params, agent_state, timestep, rng)

        value = discretizer.logits_to_scalar(preds.value_logits)

        root = mctx.RootFnOutput(
           prior_logits=preds.policy_logits,
           value=value,
           embedding=preds.state)

        # 1 step of policy improvement
        rng, improve_key = jax.random.split(rng)
        mcts_outputs = mcts_policy(
            params=train_state.params,
            rng_key=improve_key,
            root=root,
            recurrent_fn=functools.partial(
                model_step,
                discount=timestep.discount,
                agent=agent,
                discretizer=discretizer,
                evaluation=evaluation,
            ))

        policy_target = mcts_outputs.action_weights
        if evaluation:
            action = jnp.argmax(policy_target, axis=-1)
        else:
            rng, rng_ = jax.random.split(rng)
            action = jax.random.categorical(rng_, policy_target)

        return preds, action, agent_state

    return vbb.Actor(train_step=actor_step,
                     eval_step=functools.partial(
                         actor_step, evaluation=True))
