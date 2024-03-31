"""
Logic of train:
--------------

Initialize:
- environment
- agent neural network
- agent actor functions
- optimizer
- replay buffer
- observers (for logging)
- loss function


for some number of updates:

    - collect trajectory of length num_steps
    - update replay buffer with trajectory

    if env_step > learning_state:
        update learner

    periodically update target parameters
        (set by TARGET_UPDATE_INTERVAL)

    periodically log metrics from evaluation actor + learner:
        (set by LEARNER_LOG_PERIOD)

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

##############################
# Data types
##############################
Config = Dict
Action = flax.struct.PyTreeNode
Agent = nn.Module
RNGKey = jax.random.KeyArray
Params = flax.core.FrozenDict
AgentState = flax.struct.PyTreeNode
Predictions = flax.struct.PyTreeNode
Env = environment.Environment
EnvParams = environment.EnvParams
ActorStepFn = Callable[[TrainState, AgentState, TimeStep, RNGKey],
                  Tuple[Predictions, AgentState]]
EnvStepFn = Callable[[RNGKey, TimeStep, Action, EnvParams],
                       TimeStep]

class RNNInput(NamedTuple):
    obs: jax.Array
    reset: jax.Array

class Actor(NamedTuple):
  actor_step: ActorStepFn
  eval_step: ActorStepFn

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
    shared_metrics: Optional[dict] = None
    buffer_state: Optional[fbx.trajectory_buffer.TrajectoryBufferState] = None

class AcmeBatchData(flax.struct.PyTreeNode):
    timestep: TimeStep
    action: jax.Array

    @property
    def mask(self):
        return self.timestep.discount

    @property
    def discount(self):
        return self.timestep.discount

    @property
    def reward(self):
        return self.timestep.reward

class CustomTrainState(TrainState):
    target_network_params: flax.core.FrozenDict
    timesteps: int
    n_updates: int

##############################
# Loss function
##############################


def masked_mean(x, mask):
    #  if len(mask.shape) < len(x.shape):
    #    nx = len(x.shape)
    #    nd = len(mask.shape)
    #    extra = nx - nd
    #    dims = list(range(nd, nd+extra))
    #    z = jnp.multiply(x, jnp.expand_dims(mask, dims))
    #  else:
    z = jnp.multiply(x, mask)
    return (z.sum(0))/(mask.sum(0)+1e-5)

def batch_to_sequence(values: jax.Array) -> jax.Array:
    return jax.tree_map(
        lambda x: jnp.transpose(x, axes=(1, 0, *range(2, len(x.shape)))), values)

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
        _, online_state = unroll(params,
                                 online_state,
                                 burn_data.timestep)
        _, target_state = unroll(target_params,
                                 target_state,
                                 burn_data.timestep)

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

##############################
# Neural Network
##############################

class ScannedRNN(nn.Module):

    cell: nn.RNNCellBase = nn.LSTMCell

    def unroll(self, rnn_state, x: RNNInput):
        """Applies the module.

        follows example: https://flax.readthedocs.io/en/latest/api_reference/flax.linen/_autosummary/flax.linen.scan.html
        rnn_state: [B]
        x: [T, B]; however, scan over it
        """

        def body_fn(step, state, x):
          y, state = step(state, x)
          return state, y

        scan = nn.scan(
          body_fn, variable_broadcast="params",
          split_rngs={"params": False}, in_axes=0, out_axes=0)

        final_state, all_outputs = scan(self, rnn_state, x)

        return all_outputs, final_state

    @nn.compact
    def __call__(self, rnn_state, x: RNNInput):
        """Applies the module."""
        # [B, D]

        def conditional_reset(cond, init, prior):
          # [B, D]
          return jnp.where(cond[:, np.newaxis], init, prior)

        # [B, ...]
        init_state = self.initialize_carry(x.obs.shape)
        input_state = tuple(
            conditional_reset(x.reset, init, prior) for init, prior in zip(init_state, rnn_state))

        new_rnn_state, output = self.cell(input_state, x.obs)
        return output, new_rnn_state

    def initialize_carry(self, example_shape: Tuple[int]):
        return self.cell.initialize_carry(
            jax.random.PRNGKey(0), example_shape
        )


##############################
# Train loss
##############################

AgentResetFn = Callable[[Params, TimeStep], AgentState]
EnvResetFn = Callable[[RNGKey, EnvParams], TimeStep]
MakeAgentFn = Callable[[Config, Env, EnvParams, TimeStep, jax.random.KeyArray],
                       Tuple[nn.Module, Params, AgentResetFn]]
MakeOptimizerFn = Callable[[Config], optax.GradientTransformation]
MakeLossFnClass = Callable[[Config], RecurrentLossFn]
MakeActorFn = Callable[[Config, Agent], Actor]

def collect_trajectory(
      runner_state: RunnerState,
      num_steps: int,
      actor_step_fn: ActorStepFn,
      env_step_fn: EnvStepFn,
      env_params: EnvParams,
      observer: Optional[observers.BasicObserver] = None,
      ):

    def _env_step(runner_state: RunnerState, unused):
        # things that will be used/changed
        rng = runner_state.rng
        prior_timestep = runner_state.timestep
        agent_state = runner_state.agent_state
        observer_state = runner_state.observer_state

        # prepare rngs for actions and step
        rng, rng_a, rng_s = jax.random.split(rng, 3)

        preds, action, agent_state = actor_step_fn(
            runner_state.train_state,
            agent_state,
            prior_timestep,
            rng_a)

        # take step in env
        timestep = env_step_fn(rng_s, prior_timestep, action, env_params)

        # update observer with data (used for logging)
        if observer is not None:
         observer_state = observer.observe(
             observer_state=observer_state,
             next_timestep=timestep,
             predictions=preds,
             action=action)

        # create transition which will be added to buffer
        transition = Transition(
            timestep,
            action=action,
            extras=dict(preds=preds, agent_state=agent_state))

        runner_state = runner_state._replace(
            timestep=timestep,
            agent_state=agent_state,
            observer_state=observer_state,
            rng=rng,
        )

        return runner_state, transition

    return jax.lax.scan(_env_step, runner_state, None, num_steps)

def log_learner_eval(
      config: dict,
      agent_reset_fn: AgentResetFn,
      actor_train_step_fn: ActorStepFn,
      actor_eval_step_fn: ActorStepFn,
      env_reset_fn: EnvResetFn,
      env_step_fn: EnvStepFn,
      train_env_params: EnvParams,
      test_env_params: EnvParams,
      runner_state: RunnerState,
      learner_metrics: dict,
      observer: Optional[observers.BasicObserver] = None,
      observer_state: Optional[observers.BasicObserverState] = None,
      shared_metrics: dict = {},
      ):

    def log_eval(runner_state: RunnerState,
                 os: observers.BasicObserverState):
        """Help function to test greedy policy during training"""
        ########################
        # TESTING
        ########################
        # reset environment
        rng = runner_state.rng
        rng, _rng = jax.random.split(rng)
        init_timestep = env_reset_fn(_rng, test_env_params)

        # reset agent state
        init_agent_state = agent_reset_fn(
            runner_state.train_state.params,
            init_timestep)

        # new runner
        rng, _rng = jax.random.split(rng)
        eval_runner_state = RunnerState(
            train_state=runner_state.train_state,
            observer_state=observer.observe_first(
                first_timestep=init_timestep,
                observer_state=os),
            timestep=init_timestep,
            agent_state=init_agent_state,
            rng=_rng)

        final_eval_runner_state, _ = collect_trajectory(
            runner_state=eval_runner_state,
            num_steps=config["EVAL_STEPS"]*config["EVAL_EPISODES"],
            actor_step_fn=actor_eval_step_fn,
            env_step_fn=env_step_fn,
            env_params=test_env_params,
            observer=observer,
            )

        observer.flush_metrics(
            key='evaluator_performance',
            observer_state=final_eval_runner_state.observer_state,
            shared_metrics=shared_metrics,
            force=True)

        ########################
        # TRAINING
        ########################
        # reset environment
        rng = runner_state.rng
        rng, _rng = jax.random.split(rng)
        init_timestep = env_reset_fn(_rng, train_env_params)

        # reset agent state
        init_agent_state = agent_reset_fn(
            runner_state.train_state.params,
            init_timestep)
        
        # new runner
        rng, _rng = jax.random.split(rng)
        eval_runner_state = RunnerState(
            train_state=runner_state.train_state,
            observer_state=observer.observe_first(
                first_timestep=init_timestep,
                observer_state=os),
            timestep=init_timestep,
            agent_state=init_agent_state,
            rng=_rng)

        final_eval_runner_state, _ = collect_trajectory(
            runner_state=eval_runner_state,
            num_steps=config["EVAL_STEPS"]*config["EVAL_EPISODES"],
            actor_step_fn=actor_train_step_fn,
            env_step_fn=env_step_fn,
            env_params=train_env_params,
            observer=observer,
            )

        observer.flush_metrics(
            key='actor_performance',
            observer_state=final_eval_runner_state.observer_state,
            shared_metrics=shared_metrics,
            force=True)

    def log_learner(learner_metrics):
        def callback(metrics):
            if wandb.run is not None:
                wandb.log(metrics)
        jax.debug.callback(callback, learner_metrics)

    #learner_metrics.update({f'general/{k}': v for k, v in shared_metrics.items()})
    log_learner(learner_metrics)
    log_eval(runner_state, observer_state)

def make_train_step(
        config: dict,
        env: environment.Environment,
        train_env_params: environment.EnvParams,
        make_agent: MakeAgentFn,
        make_optimizer: MakeOptimizerFn,
        make_loss_fn_class: MakeLossFnClass,
        make_actor: MakeActorFn,
        test_env_params: Optional[environment.EnvParams] = None,
        ObserverCls: observers.BasicObserver = observers.BasicObserver,
        ):

    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_ENVS"]
    )

    test_env_params = test_env_params or train_env_params
    def vmap_reset(rng, env_params): 
      return jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, config["NUM_ENVS"]), env_params)

    def vmap_step(rng, env_state, action, env_params):
       return jax.vmap(
        env.step, in_axes=(0, 0, 0, None))(
           jax.random.split(rng, config["NUM_ENVS"]), env_state, action, env_params)


    def train(rng: jax.random.KeyArray):

        ##############################
        # INIT ENV
        ##############################
        rng, _rng = jax.random.split(rng)
        init_timestep = vmap_reset(_rng, train_env_params)

        ##############################
        # INIT NETWORK
        # will be absorbed into _update_step via closure
        ##############################
        rng, _rng = jax.random.split(rng)
        agent, network_params, agent_reset_fn = make_agent(
           config, env, train_env_params, init_timestep, _rng)

        init_agent_state = agent_reset_fn(network_params, init_timestep)

        ##############################
        # INIT Actor
        # will be absorbed into _update_step via closure
        ##############################
        actor = make_actor(config, agent)

        ##############################
        # INIT OPTIMIZER
        ##############################
        tx = make_optimizer(config)

        train_state = CustomTrainState.create(
            apply_fn=agent.apply,
            params=network_params,
            target_network_params=jax.tree_map(
                lambda x: jnp.copy(x), network_params),
            tx=tx,
            timesteps=0,
            n_updates=0,
        )

        ##############################
        # INIT BUFFER
        ##############################
        period = config.get("SAMPLING_PERIOD", 1)
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
        dummy_rng = jax.random.PRNGKey(0)
        init_preds, action, _ = actor.actor_step(
            train_state, init_agent_state, init_timestep, dummy_rng)
        init_transition = Transition(
            init_timestep,
            action=action,
            extras=dict(preds=init_preds, agent_state=init_agent_state))
        init_transition_example = jax.tree_map(
          lambda x: x[0], init_transition)

        # [num_envs, max_length, ...]
        buffer_state = buffer.init(init_transition_example) 

        buffer_addition = jax.tree_map(
            lambda x: x[:, np.newaxis], init_transition)
        buffer_state = buffer.add(
            buffer_state, buffer_addition)

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
        # INIT LOSS FN
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
        loss_name = config.get('LOSS_NAME', loss_name)
        def update_loss_metrics(m, ts):
          return {f'{loss_name}/{k}': v for k,v in m.items()}

        dummy_metrics = update_loss_metrics(
          dummy_metrics, train_state)

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
            timestep = vmap_step(rng_s, prior_timestep, action, train_env_params)

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
            shared_metrics['num_actor_steps'] = train_state.timesteps

            shared_metrics['num_learner_updates'] = train_state.n_updates

            # create transition which will be added to buffer
            transition = Transition(
              timestep,
              action=action,
              extras=dict(preds=preds, agent_state=agent_state))

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
                train_state.n_updates % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda train_state: train_state.replace(
                    #target_network_params=optax.incremental_update(
                    #    train_state.params,
                    #    train_state.target_network_params,
                    #    config["TAU"],
                    #)
                   target_network_params=jax.tree_map(lambda x: jnp.copy(x), train_state.params)
                ),
                lambda train_state: train_state,
                operand=train_state,
            )

            ##############################
            # 3. Logging learner metrics + evaluation episodes
            ##############################
            log_period = max(1, int(
                    config["LEARNER_LOG_PERIOD"] // config["NUM_STEPS"] // config["NUM_ENVS"]))
            is_log_time = jnp.logical_and(
                is_learn_time, train_state.n_updates % log_period == 0
            )

            jax.lax.cond(
                is_log_time,
                lambda: log_learner_eval(
                    config=config,
                    agent_reset_fn=agent_reset_fn,
                    actor_train_step_fn=actor.actor_step,
                    actor_eval_step_fn=actor.eval_step,
                    env_reset_fn=vmap_reset,
                    env_step_fn=vmap_step,
                    train_env_params=train_env_params,
                    test_env_params=test_env_params,
                    runner_state=runner_state,
                    learner_metrics=learner_metrics,
                    observer=eval_observer,
                    observer_state=init_eval_observer_state,
                    shared_metrics=shared_metrics,
                    ),
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
        shared_metrics = {
           'num_actor_steps': 0,
           'num_learner_updates': 0,
        }
        rng, _rng = jax.random.split(rng)
        runner_state = RunnerState(
          train_state=train_state,
          observer_state=observer_state,
          buffer_state=buffer_state,
          timestep=init_timestep,
          agent_state=init_agent_state,
        #  shared_metrics=shared_metrics,
          rng=_rng)

        runner_state, _ = jax.lax.scan(
            _train_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state}

    return train

def make_train_unroll(
        config: dict,
        env: environment.Environment,
        train_env_params: environment.EnvParams,
        make_agent: MakeAgentFn,
        make_optimizer: MakeOptimizerFn,
        make_loss_fn_class: MakeLossFnClass,
        make_actor: MakeActorFn,
        test_env_params: Optional[environment.EnvParams] = None,
        ObserverCls: observers.BasicObserver = observers.BasicObserver,
        ):
    """Creates a train function that does learning after unrolling agent for K timesteps.

    Args:
        config (dict): _description_
        env (environment.Environment): _description_
        env_params (environment.EnvParams): _description_
        make_agent (MakeAgentFn): _description_
        make_optimizer (MakeOptimizerFn): _description_
        make_loss_fn_class (MakeLossFnClass): _description_
        make_actor (MakeActorFn): _description_
        test_env_params (Optional[environment.EnvParams], optional): _description_. Defaults to None.
        ObserverCls (observers.BasicObserver, optional): _description_. Defaults to observers.BasicObserver.

    Returns:
        _type_: _description_
    """

    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_ENVS"]
    )

    test_env_params = test_env_params or train_env_params

    def vmap_reset(rng, env_params):
      return jax.vmap(env.reset, in_axes=(0, None))(
          jax.random.split(rng, config["NUM_ENVS"]), env_params)

    def vmap_step(rng, env_state, action, env_params):
       return jax.vmap(
           env.step, in_axes=(0, 0, 0, None))(
           jax.random.split(rng, config["NUM_ENVS"]), env_state, action, env_params)

    def train(rng: jax.random.KeyArray):

        ##############################
        # INIT ENV
        ##############################
        rng, _rng = jax.random.split(rng)
        init_timestep = vmap_reset(_rng, train_env_params)

        ##############################
        # INIT NETWORK
        # will be absorbed into _update_step via closure
        ##############################
        rng, _rng = jax.random.split(rng)
        agent, network_params, agent_reset_fn = make_agent(
            config, env, train_env_params, init_timestep, _rng)

        init_agent_state = agent_reset_fn(network_params, init_timestep)

        ##############################
        # INIT Actor
        # will be absorbed into _update_step via closure
        ##############################
        actor = make_actor(config, agent)

        ##############################
        # INIT OPTIMIZER
        ##############################
        tx = make_optimizer(config)

        train_state = CustomTrainState.create(
            apply_fn=agent.apply,
            params=network_params,
            target_network_params=jax.tree_map(
                lambda x: jnp.copy(x), network_params),
            tx=tx,
            timesteps=0,
            n_updates=0,
        )

        ##############################
        # INIT BUFFER
        ##############################
        period = config.get("SAMPLING_PERIOD", 1)
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
        dummy_rng = jax.random.PRNGKey(0)
        init_preds, action, _ = actor.actor_step(
            train_state, init_agent_state, init_timestep, dummy_rng)
        init_transition = Transition(
            init_timestep,
            action=action,
            extras=dict(preds=init_preds, agent_state=init_agent_state))
        init_transition_example = jax.tree_map(
          lambda x: x[0], init_transition)

        # [num_envs, max_length, ...]
        buffer_state = buffer.init(init_transition_example) 

        buffer_addition = jax.tree_map(
            lambda x: x[:, np.newaxis], init_transition)
        buffer_state = buffer.add(
            buffer_state, buffer_addition)

        ##############################
        # INIT Observers
        ##############################
        observer = ObserverCls(
            num_envs=config['NUM_ENVS'],
            log_period=int(config['ACTOR_LOG_PERIOD']//config['NUM_ENVS']))
        eval_observer = ObserverCls(
            num_envs=config['NUM_ENVS'],
            log_period=config['EVAL_EPISODES'])

        example_predictions, _ = agent.apply(
            train_state.params, init_agent_state, init_timestep)

        init_actor_observer_state = observer.init(
            example_timestep=init_timestep,
            example_action=action,
            example_predictions=example_predictions)

        init_eval_observer_state = eval_observer.init(
            example_timestep=init_timestep,
            example_action=action,
            example_predictions=example_predictions)

        actor_observer_state = observer.observe_first(
            first_timestep=init_timestep,
            observer_state=init_actor_observer_state)

        ##############################
        # INIT LOSS FN
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
        dummy_metrics = jax.tree_map(lambda x: x*0.0, dummy_metrics)
        loss_name = loss_fn.__class__.__name__
        loss_name = config.get('LOSS_NAME', loss_name)
        def update_loss_metrics(m, ts):
          return {f'{loss_name}/{k}': v for k, v in m.items()}

        dummy_metrics = update_loss_metrics(
            dummy_metrics, train_state)

        ##############################
        # DEFINE TRAINING LOOP
        ##############################
        def _train_step(runner_state: RunnerState, unused):
            del unused

            ##############################
            # 1. unroll for K steps + add to buffer
            ##############################
            runner_state, traj_batch = collect_trajectory(
                runner_state=runner_state,
                num_steps=config["NUM_STEPS"],
                actor_step_fn=actor.actor_step,
                env_step_fn=vmap_step,
                env_params=train_env_params)

            # things that will be used/changed
            rng = runner_state.rng
            buffer_state = runner_state.buffer_state
            train_state = runner_state.train_state
            buffer_state = runner_state.buffer_state
            shared_metrics = runner_state.shared_metrics

            # update timesteps count
            timesteps = train_state.timesteps + config["NUM_ENVS"]*config["NUM_STEPS"]
            shared_metrics['num_actor_steps'] = timesteps

            train_state = train_state.replace(timesteps=timesteps)

            num_steps, num_envs = traj_batch.timestep.reward.shape
            assert num_steps == config["NUM_STEPS"]
            assert num_envs == config["NUM_ENVS"]
            # [num_steps, num_envs, ...] -> [num_envs, num_steps, ...]
            buffer_traj_batch = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(x, 0, 1),
                traj_batch
            )

            # update buffer with data of size 
            buffer_state = buffer.add(buffer_state, buffer_traj_batch)
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
                train_state = train_state.replace(
                    n_updates=train_state.n_updates + 1)
                return train_state, metrics

            is_learn_time = (
                (buffer.can_sample(buffer_state))
                & (  # enough experience in buffer
                    timesteps >= config["LEARNING_STARTS"]
                ))

            rng, _rng = jax.random.split(rng)
            train_state, learner_metrics = jax.lax.cond(
                is_learn_time,
                lambda train_state, rng: _learn_phase(train_state, rng),
                lambda train_state, rng: (
                    train_state, dummy_metrics),  # do nothing
                train_state,
                _rng,
            )
            shared_metrics['num_learner_updates'] = train_state.n_updates

            # update target network
            train_state = jax.lax.cond(
                train_state.n_updates % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda train_state: train_state.replace(
                    target_network_params=jax.tree_map(lambda x: jnp.copy(x), train_state.params)
                ),
                lambda train_state: train_state,
                operand=train_state,
            )

            ##############################
            # 3. Creat next runner state
            ##############################
            next_runner_state = runner_state._replace(
                train_state=train_state,
                buffer_state=buffer_state,
                shared_metrics=shared_metrics,
                rng=rng)

            ##############################
            # 4. Logging learner metrics + evaluation episodes
            ##############################
            log_period = max(1, int(
                    config["LEARNER_LOG_PERIOD"] // config["NUM_STEPS"] // config["NUM_ENVS"]))
            is_log_time = jnp.logical_and(
                is_learn_time, train_state.n_updates % log_period == 0
            )

            jax.lax.cond(
                is_log_time,
                lambda: log_learner_eval(
                    config=config,
                    agent_reset_fn=agent_reset_fn,
                    actor_train_step_fn=actor.actor_step,
                    actor_eval_step_fn=actor.eval_step,
                    env_reset_fn=vmap_reset,
                    env_step_fn=vmap_step,
                    train_env_params=train_env_params,
                    test_env_params=test_env_params,
                    runner_state=runner_state,
                    learner_metrics=learner_metrics,
                    observer=eval_observer,
                    observer_state=init_eval_observer_state,
                    shared_metrics=shared_metrics,
                    ),
                lambda: None,
            )
            return next_runner_state, {}

        ##############################
        # TRAINING LOOP DEFINED. NOW RUN
        ##############################
        shared_metrics = {
           'num_actor_steps': 0,
           'num_learner_updates': 0,
        }
        # run loop
        rng, _rng = jax.random.split(rng)
        runner_state = RunnerState(
            train_state=train_state,
            observer_state=actor_observer_state,
            buffer_state=buffer_state,
            timestep=init_timestep,
            agent_state=init_agent_state,
            shared_metrics=shared_metrics,
            rng=_rng)

        runner_state, _ = jax.lax.scan(
            _train_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state}

    return train

def make_train(*args, train_step_type: str = 'unroll', **kwargs):
   if train_step_type == 'unroll':
    return make_train_unroll(*args, **kwargs)
   elif train_step_type == 'single_step':
    return make_train_step(*args, **kwargs)
   else:
      raise NotImplementedError(train_step_type)