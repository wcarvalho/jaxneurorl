"""
Alpha-Zero
"""
import functools
from typing import Callable, Dict, List, Optional, Tuple, Union

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
AgentState = flax.struct.PyTreeNode


@struct.dataclass
class Predictions:
    policy_logits: jax.Array
    value_logits: jax.Array
    state: jax.Array

@struct.dataclass
class AlphaZeroLossFn(vbb.RecurrentLossFn):
    """Computes AlphaZero loss. 
    """
    discretizer: utils.Discretizer = utils.Discretizer(
      max_value=10, num_bins=101)
    mcts_policy: Union[
      mctx.muzero_policy,
      mctx.gumbel_muzero_policy] = mctx.gumbel_muzero_policy

    simulation_steps : int = 5  # how many time-steps of simulation to learn model with
    #reanalyze_ratio : float = .5  # how often to learn from MCTS data vs. experience
    value_target_source: str = 'return'

    root_policy_coef: float = 1.0
    root_value_coef: float = 0.25

    state_from_preds: Callable[
      [Predictions], jax.Array] = lambda preds: preds.state

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
        _, rnn_state = self.__call__(rnn_state, x, rng)
        dummy_action = jnp.zeros(batch_dims, dtype=jnp.uint32)
        self.apply_model(rnn_state, dummy_action)

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

        import ipdb; ipdb.set_trace()
        policy_logits = self.policy_fn(rnn_out)
        value_logits = self.value_fn(rnn_out)
        predictions = Predictions(policy_logits, value_logits, rnn_out)

        return predictions, new_rnn_state

    def unroll(self, rnn_state, xs: TimeStep, rng: jax.random.KeyArray):
        # rnn_state: [B]
        # xs: [T, B]
        xs = extract_timestep_input(xs)

        embedding = self.observation_encoder(xs.obs)
        embedding = nn.relu(embedding)

        rnn_in = xs._replace(obs=embedding)
        rng, _rng = jax.random.split(rng)
        new_rnn_state, rnn_out = self.rnn.unroll(rnn_state, rnn_in, _rng)

        import ipdb; ipdb.set_trace()
        policy_logits = self.policy_fn(rnn_out)
        value_logits = self.value_fn(rnn_out)
        predictions = Predictions(policy_logits, value_logits, rnn_out)

        return predictions, new_rnn_state

    def apply_model(
          self,
          state: AgentState,  # [B, D]
          action: jnp.ndarray,  # [B]
      ) -> Tuple[ModelOutput, State]:
        """This applies the model to each element in the state, action vectors.

        Args:
            state (State): states. [B, D]
            action (jnp.ndarray): actions to take on states. [B]

        Returns:
            Tuple[ModelOutput, State]: muzero outputs and new states for 
              each state state action pair.
        """
        # [B, D], [B, D]
        hidden, new_state = self._transition_fn(action, state)
        model_output = self._model_pred_fn(hidden)
        return model_output, new_state


def make_agent(
        config: dict,
        env: environment.Environment,
        env_params: environment.EnvParams,
        example_timestep: TimeStep,
        rng: jax.random.KeyArray) -> Tuple[nn.Module, Params, vbb.AgentResetFn]:

    agent = AlphaZeroAgent(
        action_dim=env.action_space(env_params).n,
        hidden_dim=config["AGENT_HIDDEN_DIM"],
        init_scale=config['AGENT_INIT_SCALE'],
        rnn=vbb.ScannedRNN(hidden_dim=config["AGENT_HIDDEN_DIM"]),
        env=env,
        env_params=env_params,
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


def make_loss_fn_class(config) -> vbb.RecurrentLossFn:
  import ipdb; ipdb.set_trace()
  return functools.partial(
     AlphaZeroLossFn,
     discount=config['GAMMA'])

def make_actor(config: dict, agent: nn.Module, rng: jax.random.KeyArray) -> vbb.Actor:
    import ipdb; ipdb.set_trace()