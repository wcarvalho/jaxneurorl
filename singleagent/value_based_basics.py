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

TODO: 
- add priorizied replay
"""


from pprint import pprint
import collections
import copy
import functools
import os
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple, Dict, Optional, Tuple, Callable, TypeVar
import tree


import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.core import FrozenDict

import flashbax as fbx
import wandb

import flax
import rlax
from gymnax.environments import environment

from singleagent.basics import TimeStep


os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

from library import observers
from library import loggers

##############################
# Data types
##############################
Config = Dict
Action = flax.struct.PyTreeNode
Agent = nn.Module
PRNGKey = jax.random.KeyArray
Params = flax.core.FrozenDict
AgentState = flax.struct.PyTreeNode
Predictions = flax.struct.PyTreeNode
Env = environment.Environment
EnvParams = environment.EnvParams


ActorStepFn = Callable[[TrainState, AgentState, TimeStep, PRNGKey],
                  Tuple[Predictions, AgentState]]
EnvStepFn = Callable[[PRNGKey, TimeStep, Action, EnvParams],
                       TimeStep]

class RNNInput(NamedTuple):
    obs: jax.Array
    reset: jax.Array

class Actor(NamedTuple):
  train_step: ActorStepFn
  eval_step: ActorStepFn

class Transition(NamedTuple):
    timestep: TimeStep
    action: jax.Array
    extras: Optional[FrozenDict[str, jax.Array]] = None

class RunnerState(NamedTuple):
    train_state: TrainState
    observer_state: flax.struct.PyTreeNode
    timestep: TimeStep
    agent_state: jax.Array
    rng: jax.random.KeyArray
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
    z = jnp.multiply(x, mask)
    return (z.sum(0))/(mask.sum(0)+1e-5)

def batch_to_sequence(values: jax.Array) -> jax.Array:
    return jax.tree_map(
        lambda x: jnp.transpose(x, axes=(1, 0, *range(2, len(x.shape)))), values)

@struct.dataclass
class RecurrentLossFn:
  """Recurrent loss function with burn-in structured modelled after R2D2.
  
  https://openreview.net/forum?id=r1lyTjAqYX
  """

  network: nn.Module
  discount: float = 0.99
  tx_pair: rlax.TxPair = rlax.IDENTITY_PAIR
  burn_in_length: int = None

  data_wrapper: flax.struct.PyTreeNode = AcmeBatchData
  logger: loggers.Logger = loggers.Logger

  def __call__(
      self,
      params: Params,
      target_params: Params,
      batch: fbx.trajectory_buffer.BufferSample,
      key_grad: PRNGKey,
      steps: int,
    ):
    """Calculate a loss on a single batch of data."""
    unroll = functools.partial(self.network.apply, method=self.network.unroll)

    # Get core state & warm it up on observations for a burn-in period.
    # Replay core state.
    # [B, T, D]
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
        key_grad, rng_1, rng_2 = jax.random.split(key_grad, 3)
        _, online_state = unroll(params,
                                 online_state,
                                 burn_data.timestep,
                                 rng_1)
        _, target_state = unroll(target_params,
                                 target_state,
                                 burn_data.timestep,
                                 rng_2)

        # Only get data to learn on from after the end of the burn in period.
        data = jax.tree_map(lambda seq: seq[burn_in_length:], data)

    #--------------------------
    # Unroll on sequences to get online and target Q-Values.
    #--------------------------
    key_grad, rng_1, rng_2 = jax.random.split(key_grad, 3)
    online_preds, online_state = unroll(
        params, online_state, data.timestep, rng_1)
    target_preds, target_state = unroll(
        target_params, target_state, data.timestep, rng_2)

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

    batch_loss = batch_loss.mean()
    # TODO: support for prioritized replay

    return batch_loss, metrics

##############################
# Neural Network
##############################

class RlRnnCell(nn.Module):
    hidden_dim: int
    cell_type: str = "LSTMCell"

    def setup(self):
        cell_constructor = getattr(nn, self.cell_type)
        self.cell = cell_constructor(self.hidden_dim)

    def __call__(
        self,
        state: struct.PyTreeNode,
        x: jax.Array,
        reset: jax.Array,
        rng: PRNGKey,
        ):
        """Applies the module."""
        # [B, D]

        def conditional_reset(cond, init, prior):
          # [B, D]
          return jnp.where(cond[:, np.newaxis], init, prior)

        # [B, ...]
        init_state = self.initialize_carry(
           rng=rng,
           batch_dims=x.shape[:-1])
        input_state = tuple(
            conditional_reset(reset, init, prior) for init, prior in zip(init_state, state))

        return self.cell(input_state, x)

    def initialize_carry(
        self, rng: PRNGKey, batch_dims: Tuple[int, ...]
    ) -> Tuple[jax.Array, jax.Array]:
        """Initialize the RNN cell carry.

        Args:
        rng: random number generator passed to the init_fn.
        input_shape: a tuple providing the shape of the input to the cell.
        Returns:
        An initialized carry for the given RNN cell.
        """
        key1, key2 = jax.random.split(rng)
        mem_shape = batch_dims + (self.hidden_dim,)
        c = self.cell.carry_init(key1, mem_shape, self.cell.param_dtype)
        h = self.cell.carry_init(key2, mem_shape, self.cell.param_dtype)
        return (c, h)


class ScannedRNN(nn.Module):
    hidden_dim: int
    cell_type: str = "LSTMCell"

    def initialize_carry(self, *args, **kwargs):
        """Initializes the RNN state."""
        return self.cell.initialize_carry(*args, **kwargs)

    def setup(self):
        self.cell = RlRnnCell(cell_type=self.cell_type,
                              hidden_dim=self.hidden_dim)

    def __call__(self, state, x: RNNInput, rng: PRNGKey):
        """Applies the module.

        rnn_state: [B]
        x: [B]

        """
        return self.cell(state=state, x=x.obs, reset=x.reset, rng=rng)

    def unroll(self, state, xs: RNNInput, rng: PRNGKey):
        """
        rnn_state: [B]
        x: [T, B]; however, scan over it
        """
        cell = self.cell
        def body_fn(cell, state, inputs):
            x, reset = inputs
            return cell(state, x, reset, rng)

        scan = nn.scan(
            body_fn,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0
        )

        return scan(cell, state, (xs.obs, xs.reset))



class DummyRNN(nn.Module):
    hidden_dim: int = 0
    cell_type: str = "LSTMCell"

    def __call__(self, state, x: RNNInput, rng: PRNGKey):
       return state, x.obs

    def unroll(self, state, xs: RNNInput, rng: PRNGKey):
       return state, xs.obs

    def initialize_carry(
        self, rng: PRNGKey, batch_dims: Tuple[int, ...]
    ) -> Tuple[jax.Array, jax.Array]:
        del rng
        mem_shape = batch_dims + (self.hidden_dim,)
        return jnp.zeros(mem_shape), jnp.zeros(mem_shape)

##############################
# Train loss
##############################

AgentResetFn = Callable[[Params, TimeStep], AgentState]
EnvResetFn = Callable[[PRNGKey, EnvParams], TimeStep]
MakeAgentFn = Callable[[Config, Env, EnvParams, TimeStep, jax.random.KeyArray],
                       Tuple[nn.Module, Params, AgentResetFn]]
MakeOptimizerFn = Callable[[Config], optax.GradientTransformation]
MakeLossFnClass = Callable[[Config], RecurrentLossFn]
MakeActorFn = Callable[[Config, Agent], Actor]
MakeLoggerFn = Callable[[Config, Env, EnvParams, Agent], loggers.Logger]

def log_params(params):
    size_of_tree = lambda t: sum(tree.flatten(t))
    # Log how many parameters the network has.
    sizes = tree.map_structure(jnp.size, params)
    total_params =  size_of_tree(sizes.values())
    print("="*50)
    print(f'Total number of params: {total_params:,}')
    [print(f"\t{k}: {size_of_tree(v.values()):,}") for k,v in sizes.items()]
    pprint(jax.tree_map(lambda x:x.shape, params))

def collect_trajectory(
      runner_state: RunnerState,
      num_steps: int,
      actor_step_fn: ActorStepFn,
      env_step_fn: EnvStepFn,
      env_params: EnvParams,
      observer: Optional[observers.BasicObserver] = None,
      ):

    def _env_step(rs: RunnerState, unused):
        """_summary_
        
        Buffer is updated with:
        - input agent state: s_{t-1}
        - agent obs input: x_t
        - agent prediction outputs: p_t
        - agent's action: a_t

        Args:
            rs (RunnerState): _description_
            unused (_type_): _description_

        Returns:
            _type_: _description_
        """
        del unused
        # things that will be used/changed
        rng = rs.rng
        prior_timestep = rs.timestep
        prior_agent_state = rs.agent_state
        observer_state = rs.observer_state

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

    return jax.lax.scan(f=_env_step, init=runner_state, xs=None, length=num_steps)


def learn_step(
        train_state: CustomTrainState,
        rng: jax.random.KeyArray,
        buffer,
        buffer_state,
        loss_fn,
        ):

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

    train_state = train_state.apply_gradients(grads=grads)
    train_state = train_state.replace(n_updates=train_state.n_updates + 1)

    metrics.update({
        '0.grad_norm': optax.global_norm(grads),
        '0.param_norm': optax.global_norm(train_state.params),
    })

    return train_state, metrics, grads

def log_performance(
      config: dict,
      agent_reset_fn: AgentResetFn,
      actor_train_step_fn: ActorStepFn,
      actor_eval_step_fn: ActorStepFn,
      env_reset_fn: EnvResetFn,
      env_step_fn: EnvStepFn,
      train_env_params: EnvParams,
      test_env_params: EnvParams,
      runner_state: RunnerState,
      logger: loggers.Logger,
      observer: Optional[observers.BasicObserver] = None,
      observer_state: Optional[observers.BasicObserverState] = None,
      ):

    ########################
    # TESTING PERFORMANCE
    ########################
    # reset environment
    rng = runner_state.rng
    rng, _rng = jax.random.split(rng)
    init_timestep = env_reset_fn(_rng, test_env_params)

    # reset agent state
    rng, _rng = jax.random.split(rng)
    init_agent_state = agent_reset_fn(
        runner_state.train_state.params,
        init_timestep,
        _rng)

    # new runner
    rng, _rng = jax.random.split(rng)
    eval_runner_state = RunnerState(
        train_state=runner_state.train_state,
        observer_state=observer.observe_first(
            first_timestep=init_timestep,
            observer_state=observer_state),
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

    logger.experience_logger(
        runner_state.train_state,
        final_eval_runner_state.observer_state,
        'evaluator_performance')

    ########################
    # TRAINING PERFORMANCE
    ########################
    # reset environment
    rng = runner_state.rng
    rng, _rng = jax.random.split(rng)
    init_timestep = env_reset_fn(_rng, train_env_params)

    # reset agent state
    rng, _rng = jax.random.split(rng)
    init_agent_state = agent_reset_fn(
        runner_state.train_state.params,
        init_timestep,
        _rng)
    
    # new runner
    rng, _rng = jax.random.split(rng)
    eval_runner_state = RunnerState(
        train_state=runner_state.train_state,
        observer_state=observer.observe_first(
            first_timestep=init_timestep,
            observer_state=observer_state),
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

    logger.experience_logger(
        runner_state.train_state,
        final_eval_runner_state.observer_state,
        'actor_performance'
    )

def make_train_step(
        config: dict,
        env: environment.Environment,
        train_env_params: environment.EnvParams,
        make_agent: MakeAgentFn,
        make_optimizer: MakeOptimizerFn,
        make_loss_fn_class: MakeLossFnClass,
        make_actor: MakeActorFn,
        make_logger: MakeLoggerFn = loggers.default_make_logger,
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

        logger = make_logger(config, env, train_env_params)

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
        log_params(network_params['params'])

        rng, _rng = jax.random.split(rng)
        init_agent_state = agent_reset_fn(network_params, init_timestep, _rng)

        ##############################
        # INIT Actor
        # will be absorbed into _update_step via closure
        ##############################
        rng, _rng = jax.random.split(rng)
        actor = make_actor(config, agent, _rng)

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
            min_length_time_axis=config['SAMPLE_LENGTH'],
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
        init_preds, action, _ = actor.train_step(
            train_state, init_agent_state, init_timestep, dummy_rng)
        init_transition = Transition(
            init_timestep,
            action=action,
            extras=FrozenDict(preds=init_preds, agent_state=init_agent_state))
        init_transition_example = jax.tree_map(
          lambda x: x[0], init_transition)

        # [num_envs, max_length, ...]
        buffer_state = buffer.init(init_transition_example) 

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
            train_state.params, init_agent_state, init_timestep, dummy_rng)

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
        loss_fn = loss_fn_class(network=agent, logger=logger)

        dummy_rng = jax.random.PRNGKey(0)

        # need this for when don't compute loss metrics later on
        # this happens when you skip doing learning updates
        _, dummy_metrics, dummy_grads = learn_step(
            train_state=train_state,
            rng=dummy_rng,
            buffer=buffer,
            buffer_state=buffer_state,
            loss_fn=loss_fn)

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
            prior_agent_state = runner_state.agent_state
            buffer_state = runner_state.buffer_state

            ##############################
            # 1. env step + add to buffer
            ##############################
            # prepare rngs for actions and step
            rng, rng_a, rng_s = jax.random.split(rng, 3)

            preds, action, agent_state = actor.train_step(
               train_state,
               prior_agent_state,
               prior_timestep,
               rng_a)

            # create transition which will be added to buffer
            transition = Transition(
              prior_timestep,
              action=action,
              extras=FrozenDict(preds=preds, agent_state=prior_agent_state))


            # take step in env
            timestep = vmap_step(rng_s, prior_timestep, action, train_env_params)

            # update timesteps count
            train_state = train_state.replace(
                timesteps=train_state.timesteps + config["NUM_ENVS"]
            )


            # update buffer with data of size [num_envs, 1, ...]
            buffer_addition = jax.tree_map(
                lambda x: x[:, np.newaxis], transition)
            buffer_state = buffer.add(buffer_state, buffer_addition)

            ##############################
            # 2. Learner update
            ##############################
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

            train_state, learner_metrics, grads = jax.lax.cond(
                is_learn_time,
                lambda train_state_, rng_: learn_step(
                    train_state=train_state_,
                    rng=rng_,
                    buffer=buffer,
                    buffer_state=buffer_state,
                    loss_fn=loss_fn),
                # do nothing
                lambda train_state_, rng_: (train_state_, dummy_metrics, dummy_grads),
                train_state,
                _rng,
            )

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
            # 3. Logging learner metrics + evaluation episodes
            ##############################
            # ------------------------
            # log performance information
            # ------------------------
            log_period = max(1, int(
                    config["LEARNER_LOG_PERIOD"] // config["NUM_STEPS"] // config["NUM_ENVS"]))
            is_log_time = jnp.logical_and(
                is_learn_time, train_state.n_updates % log_period == 0
            )

            jax.lax.cond(
                is_log_time,
                lambda: log_performance(
                    config=config,
                    agent_reset_fn=agent_reset_fn,
                    actor_train_step_fn=actor.train_step,
                    actor_eval_step_fn=actor.eval_step,
                    env_reset_fn=vmap_reset,
                    env_step_fn=vmap_step,
                    train_env_params=train_env_params,
                    test_env_params=test_env_params,
                    runner_state=runner_state,
                    learner_metrics=learner_metrics,
                    observer=eval_observer,
                    observer_state=init_eval_observer_state,
                    logger=logger,
                    ),
                lambda: None,
            )

            #------------------------
            # log learner information
            # ------------------------
            loss_name = loss_fn.__class__.__name__
            jax.lax.cond(
                is_log_time,
                lambda: logger.learner_logger(
                    runner_state.train_state, learner_metrics, key=loss_name),
                lambda: None,
            )

            # ------------------------
            # log gradient information
            # ------------------------
            log_period = max(1, int(
                config.get("GRADIENT_LOG_PERIOD", 50_000) // config["NUM_STEPS"] // config["NUM_ENVS"]))
            is_log_time = train_state.n_updates % log_period == 0

            jax.lax.cond(
                is_log_time,
                lambda: logger.gradient_logger(train_state, grads),
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

def make_train_unroll(
        config: dict,
        env: environment.Environment,
        train_env_params: environment.EnvParams,
        make_agent: MakeAgentFn,
        make_optimizer: MakeOptimizerFn,
        make_loss_fn_class: MakeLossFnClass,
        make_actor: MakeActorFn,
        make_logger: MakeLoggerFn = loggers.default_make_logger,
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

    test_env_params = test_env_params or copy.deepcopy(train_env_params)

    def vmap_reset(rng, env_params):
      return jax.vmap(env.reset, in_axes=(0, None))(
          jax.random.split(rng, config["NUM_ENVS"]), env_params)

    def vmap_step(rng, env_state, action, env_params):
       return jax.vmap(
           env.step, in_axes=(0, 0, 0, None))(
           jax.random.split(rng, config["NUM_ENVS"]), env_state, action, env_params)

    def train(rng: jax.random.KeyArray):

        logger = make_logger(config, env, train_env_params)

        ##############################
        # INIT ENV
        ##############################
        rng, _rng = jax.random.split(rng)
        init_timestep = vmap_reset(rng=_rng, env_params=train_env_params)

        ##############################
        # INIT NETWORK
        # will be absorbed into _update_step via closure
        ##############################
        rng, _rng = jax.random.split(rng)
        agent, network_params, agent_reset_fn = make_agent(
            config=config,
            env=env,
            env_params=train_env_params,
            example_timestep=init_timestep,
            rng=_rng)

        log_params(network_params['params'])

        rng, _rng = jax.random.split(rng)
        init_agent_state = agent_reset_fn(network_params, init_timestep, _rng)

        ##############################
        # INIT Actor
        # will be absorbed into _update_step via closure
        ##############################
        rng, _rng = jax.random.split(rng)
        actor = make_actor(
           config=config,
           agent=agent,
           rng=_rng)

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
            min_length_time_axis=config['SAMPLE_LENGTH'],
            add_batch_size=config['NUM_ENVS'],
            sample_batch_size=config['BUFFER_BATCH_SIZE'],
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
        init_preds, action, _ = actor.train_step(
            train_state, init_agent_state, init_timestep, dummy_rng)
        init_transition = Transition(
            init_timestep,
            action=action,
            extras=FrozenDict(preds=init_preds, agent_state=init_agent_state))
        init_transition_example = jax.tree_map(
          lambda x: x[0], init_transition)

        # [num_envs, max_length, ...]
        buffer_state = buffer.init(init_transition_example) 

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
            train_state.params, init_agent_state, init_timestep, dummy_rng)

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
        loss_fn = loss_fn_class(network=agent, logger=logger)

        dummy_rng = jax.random.PRNGKey(0)

        _, dummy_metrics, dummy_grads = learn_step(
            train_state=train_state,
            rng=dummy_rng,
            buffer=buffer,
            buffer_state=buffer_state,
            loss_fn=loss_fn)

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
                actor_step_fn=actor.train_step,
                env_step_fn=vmap_step,
                env_params=train_env_params)

            # things that will be used/changed
            rng = runner_state.rng
            buffer_state = runner_state.buffer_state
            train_state = runner_state.train_state
            buffer_state = runner_state.buffer_state
            #shared_metrics = runner_state.shared_metrics

            # update timesteps count
            timesteps = train_state.timesteps + config["NUM_ENVS"]*config["NUM_STEPS"]
            #shared_metrics['num_actor_steps'] = timesteps

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
            is_learn_time = (
                (buffer.can_sample(buffer_state))
                & (  # enough experience in buffer
                    timesteps >= config["LEARNING_STARTS"]
                ))

            rng, _rng = jax.random.split(rng)
            train_state, learner_metrics, grads = jax.lax.cond(
                is_learn_time,
                lambda train_state_, rng_: learn_step(
                    train_state=train_state_,
                    rng=rng_,
                    buffer=buffer,
                    buffer_state=buffer_state,
                    loss_fn=loss_fn),
                lambda train_state, rng: (
                    train_state, dummy_metrics, dummy_grads),  # do nothing
                train_state,
                _rng,
            )

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
                rng=rng)

            ##############################
            # 4. Logging learner metrics + evaluation episodes
            ##############################
            # ------------------------
            # log performance information
            # ------------------------
            log_period = max(1, int(
                config["LEARNER_LOG_PERIOD"] // config["NUM_STEPS"] // config["NUM_ENVS"]))
            is_log_time = jnp.logical_and(
                is_learn_time, train_state.n_updates % log_period == 0
            )

            jax.lax.cond(
                is_log_time,
                lambda: log_performance(
                    config=config,
                    agent_reset_fn=agent_reset_fn,
                    actor_train_step_fn=actor.train_step,
                    actor_eval_step_fn=actor.eval_step,
                    env_reset_fn=vmap_reset,
                    env_step_fn=vmap_step,
                    train_env_params=train_env_params,
                    test_env_params=test_env_params,
                    runner_state=runner_state,
                    observer=eval_observer,
                    observer_state=init_eval_observer_state,
                    logger=logger,
                ),
                lambda: None,
            )

            # ------------------------
            # log learner information
            # ------------------------
            loss_name = loss_fn.__class__.__name__
            jax.lax.cond(
                is_log_time,
                lambda: logger.learner_logger(
                    runner_state.train_state, learner_metrics, key=loss_name),
                lambda: None,
            )

            # ------------------------
            # log gradient information
            # ------------------------
            log_period = max(1, int(
                config.get("GRADIENT_LOG_PERIOD", 50_000) // config["NUM_STEPS"] // config["NUM_ENVS"]))
            is_log_time = train_state.n_updates % log_period == 0

            jax.lax.cond(
                is_log_time,
                lambda: logger.gradient_logger(train_state, grads),
                lambda: None,
            )

            return next_runner_state, {}

        ##############################
        # TRAINING LOOP DEFINED. NOW RUN
        ##############################
        # run loop
        rng, _rng = jax.random.split(rng)
        runner_state = RunnerState(
            train_state=train_state,
            observer_state=actor_observer_state,
            buffer_state=buffer_state,
            timestep=init_timestep,
            agent_state=init_agent_state,
            rng=_rng)

        runner_state, _ = jax.lax.scan(
            _train_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state}

    return train

def make_train(*args, train_step_type: str = 'unroll', **kwargs):
   print("="*50)
   print("\t", train_step_type)
   print("="*50)
   if train_step_type == 'unroll':
    return make_train_unroll(*args, **kwargs)
   elif train_step_type == 'single_step':
    return make_train_step(*args, **kwargs)
   else:
      raise NotImplementedError(train_step_type)