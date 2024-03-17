"""

TESTING:
python -m ipdb -c continue multiagent/iql.py DEBUG=true \
  --search=default

TESTING PARALLEL:
python multiagent/iql.py \
  WANDB_MODE=enabled \
  --parallel=sbatch \
  --debug_parallel=True \
  --search=default

RUNNING:
python multiagent/iql.py \
  WANDB_MODE=enabled \
  --parallel=sbatch \
  --search=default
"""

from absl import flags
from absl import app
# from absl import logging


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

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import library.flags
from library.wrappers import TimeStep
from library import parallel
FLAGS = flags.FLAGS


def batch_to_sequence(values: jax.Array) -> jax.Array:
    return jax.tree_map(
        lambda x: jnp.transpose(x, axes=(1, 0, *range(2, len(x.shape)))), values)

def add_time_axis(values: jax.Array) -> jax.Array:
    return jax.tree_map(
      lambda x: x[np.newaxis], values)

def make_agent_input(timestep: TimeStep):
  return RNNInput(
      obs=timestep.observation,
      done=timestep.last())


class AcmeBatchData(flax.struct.PyTreeNode):
    timestep: TimeStep
    action: jax.Array

    @property
    def discount(self): 
        return self.timestep.discount

    @property
    def reward(self): 
        return self.timestep.reward

@dataclasses.dataclass
class RecurrentLossFn:
  """Recurrent loss function with burn-in structured modelled after R2D2.
  
  https://openreview.net/forum?id=r1lyTjAqYX
  """

  network: nn.Module
  discount: float = 0.99
  tx_pair: rlax.TxPair = rlax.IDENTITY_PAIR
  burn_in_length: int = None

  data_wrapper: flax.struct.PyTreeNode = AcmeBatchData

  def __call__(
      self,
      params,
      target_params,
      batch,
      key_grad,
      steps,
    ):
    """Calculate a loss on a single batch of data."""

    unroll = functools.partial(self.network.apply, method=self.network.unroll)
    # unroll = jax.vmap(unroll, 1, 1)  # vmap over batch dimension

    # Get core state & warm it up on observations for a burn-in period.
    # Replay core state.
    online_state = batch.extras.get('agent_state')

    # get online_state from 0-th time-step
    online_state = jax.tree_map(lambda x: x[:, 0], online_state)
    target_state = online_state

    # Convert sample data to sequence-major format [T, B, ...].
    data = batch_to_sequence(batch)

    #--------------------------
    # Maybe burn the core state in.
    #--------------------------
    burn_in_length = self.burn_in_length
    if burn_in_length:

      burn_data = jax.tree_map(lambda x: x[:burn_in_length], data)
      x = make_agent_input(burn_data.timestep)

      _, online_state = unroll(params, online_state, x)
      import ipdb; ipdb.set_trace()
      _, target_state = unroll(target_params, target_state, x)

    # Only get data to learn on from after the end of the burn in period.
    data = jax.tree_map(lambda seq: seq[burn_in_length:], data)

    #--------------------------
    # Unroll on sequences to get online and target Q-Values.
    #--------------------------
    x = make_agent_input(data.timestep)

    online_preds, online_state = unroll(
        params, online_state, x)
    target_preds, target_state = unroll(
        target_params, target_state, x)

    # -----------------------
    # compute loss
    # -----------------------
    data = self.data_wrapper(data.timestep, data.action)

    # [T-1, B], [B]
    elemwise_error, batch_loss, metrics = self.error(
      data=data,
      online_preds=online_preds,
      online_state=online_state,
      target_preds=target_preds,
      target_state=target_state,
      params=params,
      target_params=target_params,
      steps=steps,
      key_grad=key_grad)

    # TODO: add priorizied replay

    Cls = lambda x: x.__class__.__name__
    batch_loss = batch_loss.mean()
    return batch_loss, {Cls(self) : metrics}

@dataclasses.dataclass
class R2D2LossFn(RecurrentLossFn):

  """Loss function of R2D2.
  
  https://openreview.net/forum?id=r1lyTjAqYX
  """

  tx_pair: rlax.TxPair = rlax.IDENTITY_PAIR
  extract_q: Callable[[jax.Array], jax.Array] = lambda preds: preds.q_vals
  bootstrap_n: int = 5

  def error(self, data, online_preds, online_state, target_preds, target_state, **kwargs):
    """R2D2 learning
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
        rewards[:-1],        # [T+1] --> [T]
        discounts[:-1])      # [T+1] --> [T]

    batch_loss = 0.5 * jnp.square(batch_td_error).mean(axis=0)

    metrics = {
        '0.q_loss': batch_loss.mean(),
        '0.q_td': jnp.abs(batch_td_error).mean(),
        'z.q_mean': self.extract_q(online_preds).mean(),
        'z.q_var': self.extract_q(online_preds).var(),
        }

    return batch_td_error, batch_loss, metrics  # [T-1, B], [B]

class RNNInput(NamedTuple):
    obs: jax.Array
    done: jax.Array

class Predictions(NamedTuple):
    q_vals: jax.Array
    rnn_states: jax.Array

class Transition(NamedTuple):
    timestep: TimeStep
    action: jax.Array
    extras: Optional[dict] = None
    info: Optional[dict] = None

class RunnerState(NamedTuple):
    train_state: TrainState
    buffer_state: fbx.trajectory_buffer.TrajectoryBufferState
    timestep: TimeStep
    agent_state: jax.Array
    rng: jax.random.KeyArray

class ScannedRNN(nn.Module):

    hidden_dim: int = 128
    cell: nn.RNNCellBase = nn.LSTMCell

    def unroll(self, rnn_state, x: RNNInput):
        """Applies the module.

        follows example: https://flax.readthedocs.io/en/latest/api_reference/flax.linen/_autosummary/flax.linen.scan.html
        rnn_state: [B]
        x: [T, B]; however, scan over it
        """

        def body_fn(step, carry, x):
          y, carry = step(carry, x)
          return carry, y

        scan = nn.scan(
          body_fn, variable_broadcast="params",
          split_rngs={"params": False}, in_axes=0, out_axes=0)

        return scan(self, rnn_state, x)

    @nn.compact
    def __call__(self, rnn_state, x: RNNInput):
        """Applies the module."""
        # [B, D]

        def conditional_update(cond, new, old):
          # [B, D]
          return jnp.where(cond[:, np.newaxis], new, old)

        # [B, ...]
        updated_rnn_state = tuple(conditional_update(x.done, new, old) for new, old in zip(self.initialize_carry((x.obs.shape)), rnn_state))

        new_rnn_state, y = self.cell(updated_rnn_state, x.obs)
        return y, new_rnn_state

    def initialize_carry(self, example_shape: Tuple[int]):
        return self.cell.initialize_carry(
            jax.random.PRNGKey(0), example_shape
        )

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

        self.rnn = ScannedRNN(
            hidden_dim=self.hidden_dim,
            cell=self.cell)

        self.q_fn = nn.Dense(self.action_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))

    def initialize(self, x: RNNInput):
        """Only used for initialization."""
        # [B, D]
        rnn_state = self.initialize_carry(x.obs.shape)
        return self.__call__(rnn_state, x)

    def __call__(self, rnn_state, x: RNNInput):

        embedding = self.observation_encoder(x.obs)
        embedding = nn.relu(embedding)

        rnn_in = x._replace(obs=embedding)
        rnn_out, new_rnn_state = self.rnn(rnn_state, rnn_in)

        q_vals = self.q_fn(rnn_out)

        return Predictions(q_vals, rnn_out), new_rnn_state

    def unroll(self, rnn_state, x: RNNInput):
        # rnn_state: [B]
        # x: [T, B]
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


def make_transition(
        timestep: TimeStep,
        agent_state: jax.Array,
        info: Optional[dict] = None):
    return Transition(timestep, extras=dict(agent_state=agent_state))

class CustomTrainState(TrainState):
    target_network_params: flax.core.FrozenDict
    timesteps: int
    n_updates: int

def make_train(config, env, env_params):

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_ENVS"]
    )

    vmap_reset = lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, config["NUM_ENVS"]), env_params
    )
    vmap_step = lambda rng, env_state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, config["NUM_ENVS"]), env_state, action, env_params)


    def train(rng):

        ##############################
        # INIT NETWORK
        # will be absorbed into _update_step via closure
        # TODO: isolate out this part
        ##############################
        # [B, ...]
        init_obs = jnp.zeros((config["NUM_ENVS"], *env.observation_space(env_params).shape))
        init_done = jnp.zeros((config["NUM_ENVS"]), dtype=bool)

        agent = AgentRNN(
            action_dim=env.action_space(env_params).n,
            hidden_dim=config["AGENT_HIDDEN_DIM"],
            init_scale=config['AGENT_INIT_SCALE'],
        )

        rng, _rng = jax.random.split(rng)
        init = RNNInput(init_obs, init_done)
        network_params = agent.init(
            _rng, init, method=agent.initialize)

        init_agent_state = agent.apply(
            network_params,
            init_obs.shape,
            method=agent.initialize_carry)


        ##############################
        # INIT ENV
        ##############################
        rng, _rng = jax.random.split(rng)
        init_timestep = vmap_reset(_rng)


        ##############################
        # INIT BUFFER
        ##############################
        # to initalize the buffer is necessary to sample a trajectory to know its strucutre
        # def _env_sample_step(timestep, unused):
        #     # use a dummy rng here
        #     rng, key_a, key_s = jax.random.split(jax.random.PRNGKey(0), 3)

        #     action = env.action_space().sample(key_a)
        #     # broadcast to number of envs
        #     action = jnp.tile(action[None], [config["NUM_ENVS"]])

        #     timestep = vmap_step(key_s, timestep, action)

        #     transition = make_transition(timestep, agent_state=init_agent_state)

        #     return timestep, transition

        # _, sample_traj = jax.lax.scan(
        #     _env_sample_step, init_timestep, None, config["NUM_STEPS"]
        # )

        # # remove the NUM_ENV dim
        # sample_traj_unbatched = jax.tree_map(lambda x: x[:, 0], sample_traj)

        period = config.get("ADDING_PERIOD", config['SAMPLE_LENGTH']-1)
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=config['BUFFER_SIZE']//config['NUM_ENVS'],
            min_length_time_axis=config['BUFFER_BATCH_SIZE'],
            sample_batch_size=config['BUFFER_BATCH_SIZE'],
            add_batch_size=config['NUM_ENVS'],
            sample_sequence_length=config['SAMPLE_LENGTH'],
            period=period,
        )
        buffer = buffer.replace(
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=0),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample),
        )

        # ---------------
        # use init_transition from 0th env to initialize buffer
        # ---------------
        action = env.action_space().sample(jax.random.PRNGKey(0))
        # broadcast to number of envs
        action = jnp.tile(action[None], [config["NUM_ENVS"]])

        init_transition = Transition(
            init_timestep,
            action=action,
            extras=dict(agent_state=init_agent_state))
        init_transition = jax.tree_map(
          lambda x: x[0], init_transition)

        # [num_envs, max_length, ...]
        buffer_state = buffer.init(init_transition) 

        ##############################
        # INIT OPTIMIZER
        # TODO: isolate out this part
        ##############################
        def linear_schedule(count):
            frac = 1.0 - (count / config["NUM_UPDATES"])
            return config["LR"] * frac

        lr = linear_schedule if config.get("LR_LINEAR_DECAY", False) else config["LR"]
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=lr, eps=config['EPS_ADAM'])
        )

        train_state = CustomTrainState.create(
            apply_fn=agent.apply,
            params=network_params,
            target_network_params=jax.tree_map(lambda x: jnp.copy(x), network_params),
            tx=tx,
            timesteps=0,
            n_updates=0,
        )

        ##############################
        # INIT LOSS FN and DUMMY METRICS
        # TODO: isolate out this part
        ##############################

        loss_fn = R2D2LossFn(
          discount=config['GAMMA'],
          network=agent)
        dummy_rng = jax.random.PRNGKey(0)

        # (batch_size, timesteps, ...)
        dummy_learn_trajectory = buffer.sample(
            buffer_state, dummy_rng).experience

        _, dummy_metrics = loss_fn(
            train_state.params,
            train_state.target_network_params,
            dummy_learn_trajectory,
            dummy_rng,
            train_state.n_updates
        )
        dummy_metrics = jax.tree_map(lambda x:x*0, dummy_metrics)

        ##############################
        # INIT EXPLORATION STRATEGY
        # will be absorbed into _update_step via closure
        ##############################
        explorer = EpsilonGreedy(
            start_e=config["EPSILON_START"],
            end_e=config["EPSILON_FINISH"],
            duration=config["EPSILON_ANNEAL_TIME"]
        )

        ##############################
        # DEFINE TRAINING LOOP
        # 1. Take a set in the environment
        # 2. Update buffer
        ##############################
        def _train_step(runner_state: RunnerState, unused):

            # things that will be used/changed
            rng = runner_state.rng
            buffer_state = runner_state.buffer_state
            train_state = runner_state.train_state
            prior_timestep = runner_state.timestep
            agent_state = runner_state.agent_state
            buffer_state = runner_state.buffer_state

            ##############################
            # 1. env step + add to buffer
            ##############################
            # prepare rngs for actions and step
            rng, rng_a, rng_s = jax.random.split(rng, 3)

            x = make_agent_input(prior_timestep)
            preds, agent_state = agent.apply(
                train_state.params, agent_state, x)

            action = explorer.choose_actions(
                preds.q_vals, train_state.timesteps, rng_a)

            # take step in env
            timestep = vmap_step(rng_s, prior_timestep, action)
            transition = Transition(
              timestep,
              action=action,
              extras=dict(agent_state=init_agent_state))

            # update timesteps count
            train_state = train_state.replace(
                timesteps=train_state.timesteps + config["NUM_ENVS"]
            )

            # update buffer with [num_envs, 1, ...]
            buffer_addition = jax.tree_map(
                lambda x: x[:, np.newaxis], transition)
            buffer_state = buffer.add(
              buffer_state, buffer_addition)

            ##############################
            # 2. Learner update
            ##############################
            def _learn_phase(train_state, rng):

                # (batch_size, timesteps, ...)
                rng, _rng = jax.random.split(rng)
                learn_trajectory = buffer.sample(buffer_state, _rng).experience

                # (batch_size, timesteps, ...)
                (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                    train_state.params,
                    train_state.target_network_params,
                    learn_trajectory,
                    rng,
                    train_state.n_updates)
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(n_updates=train_state.n_updates + 1)
                return train_state, metrics

            rng, _rng = jax.random.split(rng)
            is_learn_time = (
                (buffer.can_sample(buffer_state))
                & (  # enough experience in buffer
                    train_state.timesteps > config["LEARNING_STARTS"]
                )
                & (  # pure exploration phase ended
                    train_state.timesteps % config["TRAINING_INTERVAL"] == 0
                )  # training interval
            )
            train_state, metrics = jax.lax.cond(
                is_learn_time,
                lambda train_state, rng: _learn_phase(train_state, rng),
                lambda train_state, rng: (train_state, dummy_metrics),  # do nothing
                train_state,
                _rng,
            )

            # update target network
            train_state = jax.lax.cond(
                train_state.timesteps % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda train_state: train_state.replace(
                    target_network_params=optax.incremental_update(
                        train_state.params,
                        train_state.target_network_params,
                        config["TAU"],
                    )
                ),
                lambda train_state: train_state,
                operand=train_state,
            )

            ##############################
            # 3. Logging
            ##############################


            ##############################
            # 4. Creat next runner state
            ##############################
            runner_state = RunnerState(
                train_state=train_state,
                buffer_state=buffer_state,
                timestep=timestep,
                agent_state=agent_state,
                rng=rng)

            return runner_state, metrics

        ##############################
        # TRAINING LOOP DEFINED. NOW RUN
        ##############################
        # run loop
        rng, _rng = jax.random.split(rng)
        runner_state = RunnerState(train_state, buffer_state, init_timestep, init_agent_state, _rng)

        runner_state, metrics = jax.lax.scan(
            _train_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train

