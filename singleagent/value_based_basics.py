"""

def make_agent(
        config: dict,
        env: environment.Environment,
        env_params: environment.EnvParams,
        rng: jax.random.KeyArray,
        ) -> Tuple[Agent, Params, AgentState]:
  '''This create a neural network that defines the agent.
      It also, initializes their parameters and initial
      agent state (e.g. LSTM state)'''

def make_optimizer(config: dict) -> optax.GradientTransformation:
  '''This create the optimizer (e.g. ADAM) that will optimize 
      the neural network'''

def make_loss_fn_class(config) -> vbb.RecurrentLossFn:
  '''This create the loss function that will be used for
      learning (e.g. Q-learning)'''

def make_actor(config: dict, agent: Agent) -> :
  '''This creates an Actor with two methods actor_step and eval_step.
      actor_step is used to generate actions during training.
      eval_step is used to generate actions during evaluation.
      For example, with epsilon-greedy Q-learning actor_step 
      maybe sample using the current epsilon-value whereas
      eval_step maybe also select actions according to the
      highest Q-value'''

custom_make_train = functools.partial(
   make_train,
   make_agent=make_agent,
   make_optimizer=make_optimizer,
   make_loss_fn_class=make_loss_fn_class,
   make_actor=make_actor)
"""


import functools
import os
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import NamedTuple, Dict, Union, Optional, Tuple, Callable

import dataclasses

import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import flashbax as fbx
import wandb

import flax
import rlax
from gymnax.environments import environment

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

from library.wrappers import TimeStep
from library import observers


Config = Dict
Agent = nn.Module
Params = flax.core.FrozenDict
AgentState = flax.struct.PyTreeNode
Env = environment.Environment
EnvParams = environment.EnvParams

class RNNInput(NamedTuple):
    obs: jax.Array
    done: jax.Array

class Actor(NamedTuple):
  actor_step: jax.Array
  eval_step: jax.Array

class Transition(NamedTuple):
    timestep: TimeStep
    action: jax.Array
    extras: Optional[dict] = None
    info: Optional[dict] = None

class RunnerState(NamedTuple):
    train_state: TrainState
    observer_state: flax.struct.PyTreeNode
    timestep: TimeStep
    agent_state: jax.Array
    rng: jax.random.KeyArray
    buffer_state: Optional[fbx.trajectory_buffer.TrajectoryBufferState] = None

def batch_to_sequence(values: jax.Array) -> jax.Array:
    return jax.tree_map(
        lambda x: jnp.transpose(x, axes=(1, 0, *range(2, len(x.shape)))), values)

def maked_mean(x, mask):
  if len(mask.shape) < len(x.shape):
    nx = len(x.shape)
    nd = len(mask.shape)
    extra = nx - nd
    dims = list(range(nd, nd+extra))
    z = jnp.multiply(x, jnp.expand_dims(mask, dims))
  else:
    z = jnp.multiply(x, mask)
  return (z.sum(0))/(mask.sum(0)+1e-5)

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
      _, online_state = unroll(params, online_state, burn_data.timestep)
      _, target_state = unroll(target_params, target_state, burn_data.timestep)

    # Only get data to learn on from after the end of the burn in period.
    data = jax.tree_map(lambda seq: seq[burn_in_length:], data)

    #--------------------------
    # Unroll on sequences to get online and target Q-Values.
    #--------------------------
    online_preds, online_state = unroll(
        params, online_state, data.timestep)
    target_preds, target_state = unroll(
        target_params, target_state, data.timestep)

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

    batch_loss = batch_loss.mean()
    return batch_loss, metrics

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

class CustomTrainState(TrainState):
    target_network_params: flax.core.FrozenDict
    timesteps: int
    n_updates: int


ResetFn = Callable[[Params, TimeStep], AgentState]
MakeAgentFn = Callable[[Config, Env, EnvParams, TimeStep, jax.random.KeyArray],
                       Tuple[nn.Module, Params, ResetFn]]
MakeOptimizerFn = Callable[[Config], optax.GradientTransformation]
MakeLossFnClass = Callable[[Config], RecurrentLossFn]
MakeActorFn = Callable[[Config, Agent], Actor]

def make_train(
      config: dict,
      env: environment.Environment,
      env_params: environment.EnvParams,
      make_agent: MakeAgentFn,
      make_optimizer: MakeOptimizerFn,
      make_loss_fn_class: MakeLossFnClass,
      make_actor: MakeActorFn,
      ObserverCls: observers.BasicObserver = observers.BasicObserver,
      ):

    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_ENVS"]
    )

    vmap_reset = lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, config["NUM_ENVS"]), env_params
    )
    vmap_step = lambda rng, env_state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, config["NUM_ENVS"]), env_state, action, env_params)


    def train(rng: jax.random.KeyArray):

        ##############################
        # INIT ENV
        ##############################
        rng, _rng = jax.random.split(rng)
        init_timestep = vmap_reset(_rng)

        ##############################
        # INIT NETWORK
        # will be absorbed into _update_step via closure
        ##############################
        rng, _rng = jax.random.split(rng)
        agent, network_params, agent_reset_fn = make_agent(
           config, env, env_params, init_timestep, _rng)

        init_agent_state = agent_reset_fn(network_params, init_timestep)

        ##############################
        # INIT BUFFER
        ##############################
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
        ##############################
        tx = make_optimizer(config)

        train_state = CustomTrainState.create(
            apply_fn=agent.apply,
            params=network_params,
            target_network_params=jax.tree_map(lambda x: jnp.copy(x), network_params),
            tx=tx,
            timesteps=0,
            n_updates=0,
        )

        ##############################
        # INIT Observer
        ##############################
        observer = ObserverCls(
           num_envs=config['NUM_ENVS'],
           log_period=int(config['ACTOR_LOG_PERIOD']//config['NUM_ENVS']))
        eval_observer = ObserverCls(
            num_envs=config['NUM_ENVS'],
            log_period=config['EVAL_EPISODES'])

        example_predictions, _ = agent.apply(
            train_state.params, init_agent_state, init_timestep)

        observer_state = observer.init(
            example_timestep=init_timestep,
            example_action=action,
            example_predictions=example_predictions)

        init_eval_observer_state = eval_observer.init(
            example_timestep=init_timestep,
            example_action=action,
            example_predictions=example_predictions)

        observer_state = observer.observe_first(
            first_timestep=init_timestep,
            observer_state=observer_state)

        ##############################
        # INIT LOSS FN and DUMMY METRICS
        ##############################
        loss_fn_class = make_loss_fn_class(config)
        loss_fn = loss_fn_class(network=agent)
        dummy_rng = jax.random.PRNGKey(0)

        # (batch_size, timesteps, ...)
        dummy_learn_trajectory = buffer.sample(
            buffer_state, dummy_rng).experience

        _, dummy_metrics = loss_fn(
            train_state.params,
            train_state.target_network_params,
            dummy_learn_trajectory,
            dummy_rng,
            train_state.n_updates,
        )
        dummy_metrics = jax.tree_map(lambda x:x*0.0, dummy_metrics)
        loss_name = loss_fn.__class__.__name__
        def update_loss_metrics(m, ts):
          m.update({
            'learner_steps': ts.n_updates,
            'actor_steps': ts.timesteps})
          return {f'{loss_name}/{k}': v for k,v in m.items()}

        dummy_metrics = update_loss_metrics(
          dummy_metrics, train_state)

        ##############################
        # INIT Actor
        # will be absorbed into _update_step via closure
        ##############################
        actor = make_actor(config, agent)

        ##############################
        # DEFINE TRAINING LOOP
        ##############################
        def _train_step(runner_state: RunnerState, unused):
            del unused

            # things that will be used/changed
            rng = runner_state.rng
            buffer_state = runner_state.buffer_state
            train_state = runner_state.train_state
            observer_state = runner_state.observer_state
            prior_timestep = runner_state.timestep
            agent_state = runner_state.agent_state
            buffer_state = runner_state.buffer_state

            ##############################
            # 1. env step + add to buffer
            ##############################
            # prepare rngs for actions and step
            rng, rng_a, rng_s = jax.random.split(rng, 3)

            preds, action, agent_state = actor.actor_step(
               train_state,
               agent_state,
               prior_timestep,
               rng_a)

            # take step in env
            timestep = vmap_step(rng_s, prior_timestep, action)

            # # update observer with data (used for logging)
            # observer_state = observer.observe(
            #     observer_state=observer_state,
            #     next_timestep=timestep,
            #     predictions=preds,
            #     action=action)

            # update timesteps count
            train_state = train_state.replace(
                timesteps=train_state.timesteps + config["NUM_ENVS"]
            )

            # create transition which will be added to buffer
            transition = Transition(
              timestep,
              action=action,
              extras=dict(agent_state=agent_state))

            # update buffer with data of size [num_envs, 1, ...]
            buffer_addition = jax.tree_map(
                lambda x: x[:, np.newaxis], transition)
            buffer_state = buffer.add(
              buffer_state, buffer_addition)

            ##############################
            # 2. Learner update
            ##############################
            def _learn_phase(train_state: TrainState,
                             rng: jax.random.KeyArray):

                # (batch_size, timesteps, ...)
                rng, _rng = jax.random.split(rng)
                learn_trajectory = buffer.sample(buffer_state, _rng).experience

                # (batch_size, timesteps, ...)
                rng, _rng = jax.random.split(rng)
                (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                    train_state.params,
                    train_state.target_network_params,
                    learn_trajectory,
                    _rng,
                    train_state.n_updates)
                metrics = update_loss_metrics(metrics, train_state)

                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(n_updates=train_state.n_updates + 1)
                return train_state, metrics

            is_learn_time = (
                (buffer.can_sample(buffer_state))
                & (  # enough experience in buffer
                    train_state.timesteps > config["LEARNING_STARTS"]
                )
                & (  # pure exploration phase ended
                    train_state.timesteps % config["TRAINING_INTERVAL"] == 0
                )  # training interval
            )

            rng, _rng = jax.random.split(rng)
            train_state, learner_metrics = jax.lax.cond(
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
            # 3. Logging learner metrics + evaluation episodes
            ##############################
            def log_eval(runner_state, eval_observer_state):
                """Help function to test greedy policy during training"""
                def _greedy_env_step(rs: RunnerState, unused):
                    # things that will be used/changed
                    rng = rs.rng
                    prior_timestep = rs.timestep
                    agent_state = rs.agent_state

                    # prepare rngs for actions and step
                    rng, rng_a, rng_s = jax.random.split(rng, 3)

                    preds, action, agent_state = actor.eval_step(
                       rs.train_state,
                       agent_state,
                       prior_timestep,
                       rng_a)

                    # take step in env
                    timestep = vmap_step(rng_s, prior_timestep, action)

                    observer_state = eval_observer.observe(
                        observer_state=rs.observer_state,
                        next_timestep=timestep,
                        predictions=preds,
                        action=action,
                        maybe_flush=False,
                        )

                    rs = rs._replace(
                        agent_state=agent_state,
                        timestep=timestep,
                        observer_state=observer_state,
                        rng=rng)
                    return rs, timestep

                # reset environment
                rng = runner_state.rng
                rng, _rng = jax.random.split(rng)
                init_timestep = vmap_reset(_rng)

                # reset agent state
                init_agent_state = agent_reset_fn(network_params, init_timestep)

                # create evaluation runner for greedy eva
                # unnecessary but helps ensure don't accidentally
                # re-use training information
                rng, _rng = jax.random.split(rng)
                eval_runner_state = RunnerState(
                    train_state=runner_state.train_state,
                    observer_state=observer.observe_first(
                       first_timestep=init_timestep,
                       observer_state=eval_observer_state),
                    timestep=init_timestep,
                    agent_state=init_agent_state,
                    rng=_rng)

                eval_runner_state, _ = jax.lax.scan(
                    _greedy_env_step, eval_runner_state, None,
                    config["EVAL_STEPS"]*config["EVAL_EPISODES"]
                )

                eval_observer.flush_metrics(
                   key='evaluator',
                   observer_state=eval_runner_state.observer_state,
                   force=True)

            def log_learner(learner_metrics):
              def callback(metrics):
                  if wandb.run is not None:
                    wandb.log(metrics)
              jax.debug.callback(callback, learner_metrics)

            def log(learner_metrics, runner_state, eval_observer_state):
                log_learner(learner_metrics)
                log_eval(runner_state, eval_observer_state)

            is_log_time = jnp.logical_and(
                is_learn_time,
                train_state.n_updates  % (config["LEARNER_LOG_PERIOD"] // config["NUM_STEPS"] // config["NUM_ENVS"]) == 0
               )

            jax.lax.cond(
                is_log_time,
                lambda: log(learner_metrics, runner_state, init_eval_observer_state),
                lambda: None,
            )

            ##############################
            # 4. Creat next runner state
            ##############################
            next_runner_state = RunnerState(
                train_state=train_state,
                observer_state=observer_state,
                buffer_state=buffer_state,
                timestep=timestep,
                agent_state=agent_state,
                rng=rng)

            return next_runner_state, {}

        ##############################
        # TRAINING LOOP DEFINED. NOW RUN
        ##############################
        # run loop
        rng, _rng = jax.random.split(rng)
        runner_state = RunnerState(
          train_state=train_state,
          observer_state=observer_state,
          buffer_state=buffer_state,
          timestep=init_timestep,
          agent_state=init_agent_state,
          rng=_rng)

        runner_state, _ = jax.lax.scan(
            _train_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state}

    return train

