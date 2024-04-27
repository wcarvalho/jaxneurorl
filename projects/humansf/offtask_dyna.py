"""
Dyna with the ability to do off-task simulation.
"""

from typing import Tuple, Optional
import functools

import jax
import jax.numpy as jnp
import flax
from flax import struct
import flax.linen as nn
from gymnax.environments import environment

from library import losses

from singleagent.basics import TimeStep
from singleagent import value_based_basics as vbb
from singleagent import qlearning as base_agent
from projects.humansf import qlearning

Agent = nn.Module
Params = flax.core.FrozenDict
make_actor = base_agent.make_actor
make_optimizer = base_agent.make_optimizer
learner_log_extra = qlearning.learner_log_extra
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


def simulate_trajectory(
        agent,
    ):

    def _single_step(prior_agent_state):
        """
        1. sample action
        2. apply model
        3. predict next time-step
        """

        predictions = agent.apply(prior_agent_state, timestep, rng)
        q_values = predictions.q_vals
        # prepare rngs for actions and step
        rng, rng_a, rng_s = jax.random.split(rng, 3)

        preds, action, agent_state = actor_step_fn(
            rs.train_state,
            prior_agent_state,
            prior_timestep,
            rng_a)

        transition = Transition(
            prior_timestep,
            action=action,
            extras=FrozenDict(preds=preds, agent_state=prior_agent_state))

        # take step in env
        timestep = env_step_fn(rng_s, prior_timestep, action, env_params)

        # update observer with data (used for logging)
        if observer is not None:
         observer_state = observer.observe(
             observer_state=observer_state,
             next_timestep=timestep,
             predictions=preds,
             action=action)


        rs = rs._replace(
            timestep=timestep,
            agent_state=agent_state,
            observer_state=observer_state,
            rng=rng,
        )

        return rs, transition

    return jax.lax.scan(f=_single_step, init=runner_state, xs=None, length=num_steps)


@struct.dataclass
class OfftaskDyna(base_agent.R2D2LossFn):

  """Loss function for off-task dyna
  """

  def loss_fn(
      self,
      online_preds,
      target_preds,
      actions,
      rewards,
      discounts,
      is_last,
      is_terminal,
      loss_mask,
      lambda_,
      ):
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
    target_q_t = target_q_t*is_terminal[:-1]
    batch_td_error = target_q_t - q_t
    batch_td_error = batch_td_error*loss_mask

    # [T, B]
    batch_loss = 0.5 * jnp.square(batch_td_error)

    # [B]
    batch_loss_mean = (batch_loss*loss_mask).sum(0)

    metrics = {
        '0.q_loss': batch_loss.mean(),
        '0.q_td': jnp.abs(batch_td_error).mean(),
        '1.reward': rewards[1:].mean(),
        'z.q_mean': online_preds.q_vals.mean(),
        'z.q_var': online_preds.q_vals.var(),
        }

    return batch_td_error, batch_loss_mean, metrics


  def error(self, data, online_preds, online_state, target_preds, target_state, steps, **kwargs):

    ##################
    ## Q-learning loss on batch of data
    ##################
    #online_td_error, online_batch_loss, online_metrics = self.loss_fn(
    #  online_preds=None,
    #  target_preds=None,
    #  actions=None,
    #  rewards=None,
    #  discounts=None,
    #  is_last=None,
    #  is_terminal=None,
    #  loss_mask=None,
    #  lambda_=None,
    #  )

    #################
    # Dyna Q-learning loss over simulated data
    #################
    import ipdb; ipdb.set_trace()
    # vmap over batch axis=1
    sg = lambda x: jax.lax.stop_gradient(x)
    # get states at t=1, ....
    starting_state_online = jax.tree_map(lambda x: sg(x[:-1]), online_preds.state)
    starting_state_target = jax.tree_map(lambda x: sg(x[:-1]), target_preds.state)
    # get time-steps at t=2, ...
    timesteps = jax.tree_map(lambda x: x[1:], data.timesteps)

    dyna_td_error, dyna_batch_loss, dyna_metrics = jax.vmap(jax.vmap(
      self.dyna_loss_fn))(timesteps, starting_state_online, starting_state_target)

    def dyna_loss_fn(
        self,
        timestep: TimeStep,
        online_state: jax.Array,
        target_state: jax.Array):
      """

      Algorithm:
      - Sample G (off-task) goals
      - for each goal, sample N trajectories
      - compute loss over trajectories

      Args:
          data (vbb.AcmeBatchData): [D]
          online_state (jax.Array): [D]
          target_state (jax.Array): [D]
      """

      ################
      # Sample goals
      ################
      # [D]
      # for now, just a single off-task goal
      offtask_w = timestep.state.offtask_w
      timestep = timestep.replace(
         state=timestep.state.replace(
            task_w=offtask_w))

      ################
      # Sample trajectories
      ################
      # [G, N, ...]

      self.sample_trajectories(timestep)

      ################
      # Apply loss function to trajectories
      ################
      dyna_td_error, dyna_batch_loss, dyna_metrics = self.loss_fn(
        online_preds=None,
        target_preds=None,
        actions=None,
        rewards=None,
        discounts=None,
        is_last=None,
        is_terminal=None,
        loss_mask=None,
        lambda_=None,
        )

def make_loss_fn_class(config) -> vbb.RecurrentLossFn:
  return functools.partial(
     OfftaskDyna,
     discount=config['GAMMA'])

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


class AlphaZeroAgent(nn.Module):

    action_dim: int

    observation_encoder: nn.Module
    rnn: vbb.ScannedRNN
    env: environment.Environment
    env_params: environment.EnvParams
    test_env_params: environment.EnvParams

    num_bins: int = 101

    def setup(self):

        self.policy_fn = MLP(hidden_dim=512, num_layers=1, out_dim=self.action_dim)
        self.value_fn = MLP(hidden_dim=512, num_layers=1, out_dim=self.num_bins)

    def initialize(self, x: TimeStep):
        """Only used for initialization."""
        # [B, D]
        rng = jax.random.PRNGKey(0)
        batch_dims = (x.reward.shape[0],)
        rnn_state = self.initialize_carry(rng, batch_dims)
        predictions, rnn_state = self.__call__(rnn_state, x, rng)
        dummy_action = jnp.zeros(batch_dims, dtype=jnp.int32)

        state = jax.tree_map(lambda x: x[:, None], predictions.state)
        dummy_action = jax.tree_map(lambda x: x[:, None], dummy_action)
        jax.vmap(self.apply_model, (0,0,None), 0)(state, dummy_action, rng)

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
                rnn_state=new_rnn_state)
            )

        return predictions, new_rnn_state

    def unroll(self, rnn_state, xs: TimeStep, rng: jax.random.KeyArray) -> Tuple[Predictions, RnnState]:
        # rnn_state: [B]
        # xs: [T, B]

        embedding = jax.vmap(self.observation_encoder)(xs.observation)
        embedding = nn.relu(embedding)

        rnn_in = vbb.RNNInput(obs=embedding, reset=xs.first())
        rng, _rng = jax.random.split(rng)
        new_rnn_state, new_rnn_states = self.rnn.unroll(rnn_state, rnn_in, _rng)

        rnn_out = new_rnn_states[1]
        policy_logits = jax.vmap(self.policy_fn)(rnn_out)
        value_logits = jax.vmap(self.value_fn)(rnn_out)
        predictions = Predictions(
            policy_logits=policy_logits,
            value_logits=value_logits,
            state=AgentState(
                timestep=xs,
                rnn_state=new_rnn_states)
            )
        return predictions, new_rnn_state

    def apply_model(
          self,
          state: AgentState,
          action: jnp.ndarray,
          rng: jax.random.KeyArray,
          evaluation: bool = False,
      ) -> Tuple[Predictions, RnnState]:
        """This applies the model to each element in the state, action vectors.

        Args:
            state (State): states. [1, D]
            action (jnp.ndarray): actions to take on states. [1]

        Returns:
            Tuple[ModelOutput, State]: muzero outputs and new states for 
              each state state action pair.
        """
        assert action.shape[0] == 1, 'function only accepts batchsize=1 due to inability to vmap over environment. please use vmap to get these dimensions.'
        rng, rng_ = jax.random.split(rng)
        env_params = self.test_env_params if evaluation else self.env_params
        timestep = jax.tree_map(lambda x: x[0], state.timestep)
        next_timestep = self.env.step(rng_, timestep, action[0], env_params)
        next_timestep = jax.tree_map(lambda x: x[None], next_timestep)

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
        observation_encoder=MLP(
           hidden_dim=config["AGENT_HIDDEN_DIM"],
           num_layers=1),
        rnn=vbb.ScannedRNN(
            hidden_dim=config["AGENT_HIDDEN_DIM"],
            unroll_output_state=True),
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
