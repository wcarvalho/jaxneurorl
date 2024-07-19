"""
Follows logic of 
1. Parallelized Q-learning: https://arxiv.org/pdf/2407.04811
2. Cross-Q learning: https://openreview.net/pdf?id=PczQtTsTIX

Namely:
1. no replay buffer, just trajectory mini-batches
2. no target networks
3. assume batch norm in Q-networks

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

    periodically log metrics from evaluation actor + learner:
        (set by LEARNER_LOG_PERIOD)

"""

import copy
import functools
from functools import partial
import os
from flax import struct
import jax
import jax.numpy as jnp
from jax.nn import initializers
import numpy as np
from typing import NamedTuple, Dict, Optional, Tuple, Callable, TypeVar, Any
import tree


import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.core import FrozenDict

import flashbax as fbx

from flax.linen.normalization import _compute_stats, _normalize, _canonicalize_axes
import flax
import rlax

from gymnax.environments import environment

from agents.basics import TimeStep


os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

from library import observers
from library import loggers
from library import losses

##############################
# Data types
##############################
Config = Dict
Action = flax.struct.PyTreeNode
Agent = nn.Module
PRNGKey = jax.random.PRNGKey
Params = flax.core.FrozenDict
AgentState = flax.struct.PyTreeNode
Predictions = flax.struct.PyTreeNode
Env = environment.Environment
EnvParams = environment.EnvParams
Array = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?


ActorStepFn = Callable[[TrainState, AgentState, TimeStep, jax.random.PRNGKey],
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
    timestep: TimeStep
    agent_state: jax.Array
    rng: jax.random.PRNGKey
    observer_state: Optional[flax.struct.PyTreeNode] = None
    buffer_state: Optional[fbx.trajectory_buffer.TrajectoryBufferState] = None

class AcmeBatchData(flax.struct.PyTreeNode):
    timestep: TimeStep
    action: jax.Array
    extras: FrozenDict

    @property
    def is_last(self):
        return self.timestep.last()

    @property
    def discount(self):
        return self.timestep.discount

    @property
    def reward(self):
        return self.timestep.reward

class CustomTrainState(TrainState):
    batch_stats: Any = None
    timesteps: int = 0
    n_updates: int = 0
    n_grad_steps: int = 0
    n_logs: int = 0

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
  lambda_: float = .9
  tx_pair: rlax.TxPair = rlax.IDENTITY_PAIR
  burn_in_length: int = None

  data_wrapper: flax.struct.PyTreeNode = AcmeBatchData
  logger: loggers.Logger = loggers.Logger

  def __call__(
      self,
      params: Params,
      batch: fbx.trajectory_buffer.BufferSample,
      key_grad: jax.random.PRNGKey,
      steps: int,
    ):
    """Calculate a loss on a single batch of data."""
    unroll = functools.partial(self.network.apply, method=self.network.unroll)

    # Get core state & warm it up on observations for a burn-in period.
    # Replay core state.
    # [B, T, D]
    state = batch.extras.get('agent_state')

    # get state from 0-th time-step
    state = jax.tree_map(lambda x: x[:, 0], state)

    # Convert sample data to sequence-major format [T, B, ...].
    data = batch_to_sequence(batch)

    # Unroll on sequences to get online and target Q-Values.
    key_grad, rng = jax.random.split(key_grad)
    preds, _ = unroll(params, state, data.timestep, rng)

    # compute loss
    data = self.data_wrapper(
        timestep=data.timestep,
        action=data.action,
        extras=data.extras)

    # [T-1, B], [B]
    _, batch_loss, metrics = self.error(
      data=data,
      preds=preds,
      state=state,
      params=params,
      steps=steps,
      key_grad=key_grad)
    batch_loss = batch_loss.mean()

    return batch_loss, metrics


@struct.dataclass
class PQNLossFn(RecurrentLossFn):

  """Loss function of R2D2.
  
  https://openreview.net/forum?id=r1lyTjAqYX
  """

  tx_pair: rlax.TxPair = rlax.IDENTITY_PAIR
  extract_q: Callable[[jax.Array], jax.Array] = lambda preds: preds.q_vals

  def error(self, data, preds, steps, **kwargs):
    """R2D2 learning.
    """

    def float(x): return x.astype(jnp.float32)
    # Get value-selector actions from online Q-values for double Q-learning.
    selector_actions = jnp.argmax(
        self.extract_q(preds), axis=-1)  # [T+1, B]

    # Preprocess discounts & rewards.
    discounts = float(data.discount)*self.discount
    lambda_ = jnp.ones_like(data.discount)*self.lambda_
    rewards = float(data.reward)
    is_last = float(data.is_last)

    # Get N-step transformed TD error and loss.
    batch_td_error_fn = jax.vmap(
        losses.q_learning_lambda_td,
        in_axes=1,
        out_axes=1)

    # [T, B]
    q_t, target_q_t = batch_td_error_fn(
        self.extract_q(preds)[:-1],  # [T+1] --> [T]
        data.action[:-1],    # [T+1] --> [T]
        self.extract_q(preds)[1:],  # [T+1] --> [T]
        selector_actions[1:],  # [T+1] --> [T]
        rewards[1:],        # [T+1] --> [T]
        discounts[1:],
        is_last[1:],
        lambda_[1:])      # [T+1] --> [T]

    # ensure target = 0 when episode terminates
    target_q_t = target_q_t*data.discount[:-1]
    batch_td_error = target_q_t - q_t

    # ensure loss = 0 when episode truncates
    # truncated if FINAL time-step but data.discount = 1.0, something like [1,1,2,1,1]
    # truncated is discount on AND is last
    truncated = (data.discount+is_last) > 1
    loss_mask = (1-truncated).astype(batch_td_error.dtype)[:-1]
    batch_td_error = batch_td_error*loss_mask

    # [T, B]
    batch_loss = 0.5 * jnp.square(batch_td_error)

    # [B]
    batch_loss_mean = (batch_loss*loss_mask).mean()

    metrics = {
        '0.q_loss': batch_loss.mean(),
        '0.q_td': jnp.abs(batch_td_error).mean(),
        '1.reward': rewards[1:].mean(),
        'z.q_mean': self.extract_q(preds).mean(),
        'z.q_var': self.extract_q(preds).var(),
    }

    if self.logger.learner_log_extra is not None:
        self.logger.learner_log_extra({
            'data': data,
            'td_errors': jnp.abs(batch_td_error),                 # [T]
            'mask': loss_mask,                 # [T]
            'q_values': self.extract_q(preds),    # [T, B]
            'q_loss': batch_loss,  # [ T, B]
            'q_target': target_q_t,
            'n_updates': steps,
        })

    return batch_td_error, batch_loss_mean, metrics  # [T-1, B], [B]

def make_loss_fn_class(config) -> RecurrentLossFn:
  return functools.partial(
      PQNLossFn,
      discount=config['GAMMA'])

##############################
# Neural Network Components
##############################

class RlRnnCell(nn.Module):
    hidden_dim: int
    cell_type: str = "OptimizedLSTMCell"

    def setup(self):
        cell_constructor = getattr(nn, self.cell_type)
        self.cell = cell_constructor(self.hidden_dim)

    def __call__(
        self,
        state: struct.PyTreeNode,
        x: jax.Array,
        reset: jax.Array,
        rng: jax.random.PRNGKey,
        ):
        """Applies the module."""
        # [B, D] or [D]

        def conditional_reset(cond, init, prior):
          if cond.ndim == 1:
            # [B, D]
            return jnp.where(cond[:, np.newaxis], init, prior)
          else:
            # [D]
            return jnp.where(cond[np.newaxis], init, prior)

        # [B, ...]
        init_state = self.initialize_carry(
           rng=rng,
           batch_dims=x.shape[:-1])
        if "lstm" in self.cell_type.lower():
            input_state = tuple(
                conditional_reset(reset, init, prior) for init, prior in zip(init_state, state))
        elif 'gru' in self.cell_type.lower():
            input_state = conditional_reset(reset, init_state, state)
        else:
           raise NotImplementedError(self.cell_type)

        return self.cell(input_state, x)

    def output_from_state(self, state):
        if "lstm" in self.cell_type.lower():
            return state[1]
        elif 'gru' in self.cell_type.lower():
            return state
        else:
           raise NotImplementedError(self.cell_type)

    def initialize_carry(
        self, rng: jax.random.PRNGKey, batch_dims: Tuple[int, ...]
    ) -> Tuple[jax.Array, jax.Array]:
        """Initialize the RNN cell carry.

        Args:
        rng: random number generator passed to the init_fn.
        input_shape: a tuple providing the shape of the input to the cell.
        Returns:
        An initialized carry for the given RNN cell.
        """
        # (1,) will be ignored so doesn't matter
        return self.cell.initialize_carry(
           rng, input_shape=batch_dims + (1,))

class ScannedRNN(nn.Module):
    hidden_dim: int
    cell_type: str = "OptimizedLSTMCell"
    unroll_output_state: bool = False  # return state at all time-points

    def initialize_carry(self, *args, **kwargs):
        """Initializes the RNN state."""
        return self.cell.initialize_carry(*args, **kwargs)

    def setup(self):
        self.cell = RlRnnCell(
           cell_type=self.cell_type,
           hidden_dim=self.hidden_dim,
           name=self.cell_type)

    def __call__(self, state, x: RNNInput, rng: jax.random.PRNGKey):
        """Applies the module.

        rnn_state: [B]
        x: [B]

        """
        return self.cell(state=state, x=x.obs, reset=x.reset, rng=rng)

    def unroll(self, state, xs: RNNInput, rng: jax.random.PRNGKey):
        """
        rnn_state: [B]
        x: [T, B]; however, scan over it
        """
        def body_fn(cell, state, inputs):
            x, reset = inputs
            state, out = cell(state, x, reset, rng)
            if self.unroll_output_state:
                return state, state
            return state, out

        scan = nn.scan(
            body_fn,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0
        )

        return scan(self.cell, state, (xs.obs, xs.reset))

    def output_from_state(self, state):
        return self.cell.output_from_state(state)

class DummyRNN(nn.Module):
    hidden_dim: int = 0
    cell_type: str = "OptimizedLSTMCell"
    unroll_output_state: bool = False  # return state at all time-points

    def __call__(self, state, x: RNNInput, rng: jax.random.PRNGKey):
       return state, x.obs

    def unroll(self, state, xs: RNNInput, rng: jax.random.PRNGKey):
       if self.unroll_output_state:
           return state, (xs.obs, xs.obs)
       return state, xs.obs

    def output_from_state(self, state):
        return state

    def initialize_carry(
        self, rng: jax.random.PRNGKey, batch_dims: Tuple[int, ...]
    ) -> Tuple[jax.Array, jax.Array]:
        del rng
        mem_shape = batch_dims + (self.hidden_dim,)
        return jnp.zeros(mem_shape), jnp.zeros(mem_shape)

class BatchRenorm(nn.Module):
    """BatchRenorm Module, implemented based on the Batch Renormalization paper (https://arxiv.org/abs/1702.03275).
    and adapted from Flax's BatchNorm implementation:
    https://github.com/google/flax/blob/ce8a3c74d8d1f4a7d8f14b9fb84b2cc76d7f8dbf/flax/linen/normalization.py#L228
    """

    use_running_average: Optional[bool] = None
    axis: int = -1
    momentum: float = 0.999
    epsilon: float = 0.001
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[jax.random.PRNGKey, Shape, Dtype],
                        Array] = initializers.zeros
    scale_init: Callable[[jax.random.PRNGKey,
                          Shape, Dtype], Array] = initializers.ones
    axis_name: Optional[str] = None
    axis_index_groups: Any = None
    use_fast_variance: bool = True

    @nn.compact
    def __call__(self, x, use_running_average: Optional[bool] = None):

        use_running_average = nn.merge_param(
            "use_running_average", self.use_running_average, use_running_average
        )
        feature_axes = _canonicalize_axes(x.ndim, self.axis)
        reduction_axes = tuple(i for i in range(x.ndim)
                               if i not in feature_axes)
        feature_shape = [x.shape[ax] for ax in feature_axes]

        ra_mean = self.variable(
            "batch_stats",
            "mean",
            lambda s: jnp.zeros(s, jnp.float32),
            feature_shape,
        )
        ra_var = self.variable(
            "batch_stats", "var", lambda s: jnp.ones(
                s, jnp.float32), feature_shape
        )

        r_max = self.variable(
            "batch_stats",
            "r_max",
            lambda s: s,
            3,
        )
        d_max = self.variable(
            "batch_stats",
            "d_max",
            lambda s: s,
            5,
        )
        steps = self.variable(
            "batch_stats",
            "steps",
            lambda s: s,
            0,
        )

        if use_running_average:
            mean, var = ra_mean.value, ra_var.value
            custom_mean = mean
            custom_var = var
        else:
            mean, var = _compute_stats(
                x,
                reduction_axes,
                dtype=self.dtype,
                axis_name=self.axis_name if not self.is_initializing() else None,
                axis_index_groups=self.axis_index_groups,
                use_fast_variance=self.use_fast_variance,
            )
            custom_mean = mean
            custom_var = var
            if not self.is_initializing():
                # The code below is implemented following the Batch Renormalization paper
                r = 1
                d = 0
                std = jnp.sqrt(var + self.epsilon)
                ra_std = jnp.sqrt(ra_var.value + self.epsilon)
                r = jax.lax.stop_gradient(std / ra_std)
                r = jnp.clip(r, 1 / r_max.value, r_max.value)
                d = jax.lax.stop_gradient((mean - ra_mean.value) / ra_std)
                d = jnp.clip(d, -d_max.value, d_max.value)
                tmp_var = var / (r**2)
                tmp_mean = mean - d * jnp.sqrt(custom_var) / r

                # Warm up batch renorm for 100_000 steps to build up proper running statistics
                warmed_up = jnp.greater_equal(
                    steps.value, 1000).astype(jnp.float32)
                custom_var = warmed_up * tmp_var + \
                    (1.0 - warmed_up) * custom_var
                custom_mean = warmed_up * tmp_mean + \
                    (1.0 - warmed_up) * custom_mean

                ra_mean.value = (
                    self.momentum * ra_mean.value + (1 - self.momentum) * mean
                )
                ra_var.value = self.momentum * \
                    ra_var.value + (1 - self.momentum) * var
                steps.value += 1

        return _normalize(
            self,
            x,
            custom_mean,
            custom_var,
            reduction_axes,
            feature_axes,
            self.dtype,
            self.param_dtype,
            self.epsilon,
            self.use_bias,
            self.use_scale,
            self.bias_init,
            self.scale_init,
        )

def get_activation_fn(k: str):
    if k == 'relu': return nn.relu
    elif k == 'leaky_relu': return nn.leaky_relu
    elif k == 'tanh': return nn.tanh
    raise NotImplementedError(k)

class MLP(nn.Module):
  hidden_dim: int
  out_dim: Optional[int] = 0
  num_layers: int = 1
  norm_type: str = 'none'
  activation: str = 'relu'
  activate_final: bool = True

  @nn.compact
  def __call__(self, x, train: bool = False):
    activation_fn = get_activation_fn(self.activation)

    if self.norm_type == 'none':
        normalize = lambda x: x
    elif self.norm_type == 'layer_norm':
        normalize = lambda x: nn.LayerNorm()(x)
    elif self.norm_type == 'batch_norm':
        normalize = lambda x: BatchRenorm(use_running_average=not train)(x)
    else:
        raise NotImplementedError(self.norm_type)

    for _ in range(self.num_layers):
        x = nn.Dense(self.hidden_dim, use_bias=False)(x)
        x = normalize(x)
        x = activation_fn(x)

    if self.out_dim == 0:
        return x

    x = nn.Dense(self.out_dim or self.hidden_dim, use_bias=False)(x)

    if self.activate_final:
       x = activation_fn(x)

    return x


class QPredictions(NamedTuple):
    q_vals: jax.Array
    rnn_states: jax.Array

class RnnAgent(nn.Module):
    """_summary_

    - observation encoder: CNN
    Args:
        nn (_type_): _description_

    Returns:
        _type_: _description_
    """

    action_dim: int
    observation_encoder: nn.Module
    rnn: ScannedRNN
    norm_qfn: str = 'none'

    def setup(self):

        self.q_fn = MLP(
            hidden_dim=512,
            num_layers=1,
            norm_type=self.norm_qfn,
            out_dim=self.action_dim)

    def initialize(self, x: TimeStep):
        """Only used for initialization."""

        rng = jax.random.PRNGKey(0)
        batch_dims = (x.reward.shape[0],)
        rnn_state = self.initialize_carry(rng, batch_dims)

        return self(rnn_state, x, rng)

    def __call__(
            self,
            rnn_state,
            x: TimeStep,
            rng: jax.random.PRNGKey,
            train: bool = False,
            ):

        obs = x.observation
        embedding = self.observation_encoder(obs, train=train)

        rnn_in = RNNInput(obs=embedding, reset=x.first())
        rng, _rng = jax.random.split(rng)
        new_rnn_state, rnn_out = self.rnn(rnn_state, rnn_in, _rng)

        q_vals = self.q_fn(rnn_out)

        return QPredictions(q_vals, rnn_out), new_rnn_state

    def unroll(self, rnn_state, xs: TimeStep, rng: jax.random.PRNGKey, train: bool = True):
        # rnn_state: [B]
        # xs: [T, B]
        
        obs = xs.observation
        embedding = nn.BatchApply(partial(self.observation_encoder, train=train))(obs)

        rnn_in = RNNInput(obs=embedding, reset=xs.first())
        rng, _rng = jax.random.split(rng)
        new_rnn_state, rnn_out = self.rnn.unroll(rnn_state, rnn_in, _rng)

        q_vals = nn.BatchApply(self.q_fn)(rnn_out)

        return QPredictions(q_vals, rnn_out), new_rnn_state

    def initialize_carry(self, *args, **kwargs):
        """Initializes the RNN state."""
        return self.rnn.initialize_carry(*args, **kwargs)

##############################
# Train functions
##############################

AgentResetFn = Callable[[Params, TimeStep], AgentState]
EnvResetFn = Callable[[PRNGKey, EnvParams], TimeStep]
MakeAgentFn = Callable[[Config, Env, EnvParams, TimeStep, jax.random.PRNGKey],
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
        rng: jax.random.PRNGKey,
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
        learn_trajectory,
        _rng,
        train_state.n_grad_steps)

    train_state = train_state.apply_gradients(grads=grads)
    train_state = train_state.replace(
        n_grad_steps=train_state.n_grad_steps + 1)

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
    eval_log_period_eval = config.get("EVAL_LOG_PERIOD", 10)
    if eval_log_period_eval > 0:
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
            'evaluator_performance',
            # log trajectory details for evaluator at this period
            # counter = number of times logger
            # e.g., every 10th-log log details
            log_details_period=eval_log_period_eval,
        )

    ########################
    # TRAINING PERFORMANCE
    ########################
    # reset environment
    eval_log_period_actor = config.get("EVAL_LOG_PERIOD_ACTOR", 20)
    if eval_log_period_actor > 0:
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
            'actor_performance',
            log_details_period=config.get("EVAL_LOG_PERIOD_ACTOR", 20),
        )

def make_optimizer(config: dict) -> optax.GradientTransformation:
  lr_scheduler = optax.linear_schedule(
      init_value=config["LR"],
      end_value=1e-20,
      transition_steps=(config["NUM_UPDATES_DECAY"])
      * config["NUM_MINIBATCHES"]
      * config["NUM_EPOCHS"],
  )

  lr = lr_scheduler if config.get(
      "LR_LINEAR_DECAY", False) else config["LR"]

  return optax.chain(
      optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
      optax.adam(learning_rate=lr, eps=config['EPS_ADAM'])
  )

def make_agent(
        config: dict,
        env: environment.Environment,
        env_params: environment.EnvParams,
        example_timestep: TimeStep,
        rng: jax.random.PRNGKey,
        ObsEncoderCls: nn.Module = None,
) -> Tuple[Agent, Params, AgentResetFn]:

    cell_type = config.get('RNN_CELL_TYPE', 'OptimizedLSTMCell')
    if cell_type.lower() == 'none':
        rnn = DummyRNN()
    else:
        rnn = ScannedRNN(
            hidden_dim=config["AGENT_RNN_DIM"],
            cell_type=cell_type,
        )
    if ObsEncoderCls is None:
        ObsEncoderCls = lambda: MLP(
            hidden_dim=256,
            num_layers=3,
            norm_type='layer_norm',
            )

    agent = RnnAgent(
        observation_encoder=ObsEncoderCls(),
        action_dim=env.num_actions(env_params),
        rnn=rnn,
        norm_qfn=config.get("NORM_QFN", 'none')
    )

    rng, _rng = jax.random.split(rng)
    network_params = agent.init(
        _rng, example_timestep, method=agent.initialize)

    def reset_fn(params, example_timestep, reset_rng):
      batch_dims = (example_timestep.reward.shape[0],)
      return agent.apply(
          params,
          batch_dims=batch_dims,
          rng=reset_rng,
          method=agent.initialize_carry)

    return agent, network_params, reset_fn

def epsilon_greedy_act(q, eps, key):
    # a key for sampling random actions and one for picking
    key_a, key_e = jax.random.split(key, 2)
    greedy_actions = jnp.argmax(q, axis=-1)  # get the greedy actions
    random_actions = jax.random.randint(
        key_a, shape=greedy_actions.shape, minval=0, maxval=q.shape[-1])  # sample random actions
    # pick which actions should be random
    pick_random = jax.random.uniform(key_e, greedy_actions.shape) < eps
    chosen_actions = jnp.where(pick_random, random_actions, greedy_actions)
    return chosen_actions

class LinearDecayEpsilonGreedy:
    """Epsilon Greedy action selection"""

    def __init__(self, start_e: float, end_e: float, duration: int):
        self.start_e = start_e
        self.end_e = end_e
        self.duration = duration
        self.slope = (end_e - start_e) / duration

    @partial(jax.jit, static_argnums=0)
    def get_epsilon(self, t: int):
        e = self.slope*t + self.start_e
        return jnp.clip(e, self.end_e)

    @partial(jax.jit, static_argnums=0)
    def choose_actions(self, q_vals: jnp.ndarray, t: int, rng: jax.random.PRNGKey):

        eps = self.get_epsilon(t)
        rng = jax.random.split(rng, q_vals.shape[0])
        return jax.vmap(epsilon_greedy_act, in_axes=(0, None, 0))(q_vals, eps, rng)

class FixedEpsilonGreedy:
    """Epsilon Greedy action selection"""

    def __init__(self, epsilons: float):
        self.epsilons = epsilons

    @partial(jax.jit, static_argnums=0)
    def choose_actions(self, q_vals: jnp.ndarray, t: int, rng: jax.random.PRNGKey):

        rng = jax.random.split(rng, q_vals.shape[0])
        return jax.vmap(epsilon_greedy_act, in_axes=(0, 0, 0))(q_vals, self.epsilons, rng)

def make_actor(config: dict, agent: Agent, rng: jax.random.PRNGKey) -> Actor:
    fixed_epsilon = config.get('FIXED_EPSILON', 1)
    assert fixed_epsilon in (0, 1, 2)
    if fixed_epsilon:
        # BELOW was copied from ACME
        if fixed_epsilon == 1:
            vals = np.logspace(
                start=config.get('EPSILON_MIN', 1),
                stop=config.get('EPSILON_MAX', 3),
                num=config.get('NUM_EPSILONS', 256),
                base=config.get('EPSILON_BASE', .1))
        else:
            # BELOW is in range of ~(.9,.1)
            vals = np.logspace(
                num=config.get('NUM_EPSILONS', 256),
                start=config.get('EPSILON_MIN', .05),
                stop=config.get('EPSILON_MAX', .9),
                base=config.get('EPSILON_BASE', .1))
        epsilons = jax.random.choice(
            rng, vals, shape=(config['NUM_ENVS'],))

        explorer = FixedEpsilonGreedy(epsilons)
    else:
        explorer = LinearDecayEpsilonGreedy(
            start_e=config["EPSILON_START"],
            end_e=config["EPSILON_FINISH"],
            duration=config["EPSILON_ANNEAL_TIME"]
        )

    def actor_step(
            train_state: TrainState,
            agent_state: jax.Array,
            timestep: TimeStep,
            rng: jax.random.PRNGKey):
        preds, agent_state = agent.apply(
            train_state.params, agent_state, timestep, rng)

        action = explorer.choose_actions(
            preds.q_vals, train_state.timesteps, rng)

        return preds, action, agent_state

    def eval_step(
            train_state: TrainState,
            agent_state: jax.Array,
            timestep: TimeStep,
            rng: jax.random.PRNGKey):
        preds, agent_state = agent.apply(
            train_state.params, agent_state, timestep, rng)

        action = preds.q_vals.argmax(-1)

        return preds, action, agent_state

    return Actor(train_step=actor_step, eval_step=eval_step)

def make_train(
        config: dict,
        env: environment.Environment,
        train_env_params: environment.EnvParams,
        make_agent: MakeAgentFn = make_agent,
        make_loss_fn_class: MakeLossFnClass = make_loss_fn_class,
        make_actor: MakeActorFn = make_actor,
        make_optimizer: MakeOptimizerFn = make_optimizer,
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

    config['NUM_ENVS'] = config['BATCH_SIZE'] // config["TRAINING_INTERVAL"]

    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config['BATCH_SIZE']
    )
    config["NUM_UPDATES_DECAY"] = config["NUM_UPDATES"]

    test_env_params = test_env_params or copy.deepcopy(train_env_params)

    def vmap_reset(rng, env_params):
      return jax.vmap(env.reset, in_axes=(0, None))(
          jax.random.split(rng, config["NUM_ENVS"]), env_params)

    def vmap_step(rng, env_state, action, env_params):
       return jax.vmap(
           env.step, in_axes=(0, 0, 0, None))(
           jax.random.split(rng, config["NUM_ENVS"]), env_state, action, env_params)

    def train(rng: jax.random.PRNGKey):
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
            tx=tx,
        )

        ##############################
        # INIT BUFFER
        ##############################
        sample_batch_size = config['NUM_ENVS'] // config["NUM_MINIBATCHES"]
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=config['TRAINING_INTERVAL'],
            min_length_time_axis=config['TRAINING_INTERVAL'],
            add_batch_size=config['NUM_ENVS'],
            sample_batch_size=sample_batch_size,
            sample_sequence_length=config['TRAINING_INTERVAL'],
            period=1,
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
            log_period=config.get("OBSERVER_PERIOD", 5_000),
            max_num_episodes=config.get("OBSERVER_EPISODES", 200),
            )
        eval_observer = observer

        init_actor_observer_state = observer.init(
            example_timestep=init_timestep,
            example_action=action,
            example_predictions=init_preds)

        init_eval_observer_state = eval_observer.init(
            example_timestep=init_timestep,
            example_action=action,
            example_predictions=init_preds)

        actor_observer_state = observer.observe_first(
            first_timestep=init_timestep,
            observer_state=init_actor_observer_state)

        ##############################
        # INIT LOSS FN
        ##############################
        loss_fn_class = make_loss_fn_class(config)
        loss_fn = loss_fn_class(network=agent, logger=logger)

        ##############################
        # DEFINE TRAINING LOOP
        ##############################

        def _train_step(old_runner_state: RunnerState, unused):
            del unused

            ##############################
            # 1. unroll for K steps + add to buffer
            ##############################
            runner_state, traj_batch = collect_trajectory(
                runner_state=old_runner_state,
                num_steps=config["TRAINING_INTERVAL"],
                actor_step_fn=actor.train_step,
                env_step_fn=vmap_step,
                env_params=train_env_params)

            # things that will be used/changed
            rng = runner_state.rng
            buffer_state = runner_state.buffer_state
            train_state = runner_state.train_state
            buffer_state = runner_state.buffer_state

            # update timesteps count
            timesteps = train_state.timesteps + config["NUM_ENVS"]*config["TRAINING_INTERVAL"]

            train_state = train_state.replace(timesteps=timesteps)

            num_steps, num_envs = traj_batch.timestep.reward.shape
            assert num_steps == config["TRAINING_INTERVAL"]
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
            rng, rng_ = jax.random.split(rng)
            def _learn_epoch(train_state_, ignored):
                del ignored
                train_state_, metrics, grads = learn_step(
                    train_state=train_state_,
                    rng=rng_,
                    buffer=buffer,
                    buffer_state=buffer_state,
                    loss_fn=loss_fn)

                return train_state_, (metrics, grads)

            train_state, (learner_metrics, grads) = jax.lax.scan(
                _learn_epoch, train_state, None,
                config["NUM_EPOCHS"]*config["NUM_MINIBATCHES"]
            )

            # use only last one
            learner_metrics = jax.tree_map(lambda x:x[-1], learner_metrics)
            grads = jax.tree_map(
                lambda x: x[-1], grads)

            train_state = train_state.replace(n_updates=train_state.n_updates + 1)

            ##############################
            # 3. Logging learner metrics + evaluation episodes
            ##############################
            # ------------------------
            # log performance information
            # ------------------------
            log_period = max(1, int(config["LEARNER_LOG_PERIOD"]))
            is_log_time = train_state.n_updates % log_period == 0

            train_state = jax.lax.cond(
                is_log_time,
                lambda: train_state.replace(n_logs=train_state.n_logs + 1),
                lambda: train_state,
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
            gradient_log_period = config.get("GRADIENT_LOG_PERIOD", 500)
            if gradient_log_period:
                log_period = max(1, int(gradient_log_period))
                is_log_time = train_state.n_updates % log_period == 0

                jax.lax.cond(
                    is_log_time,
                    lambda: logger.gradient_logger(train_state, grads),
                    lambda: None,
                )

            ##############################
            # 4. Creat next runner state
            ##############################
            next_runner_state = runner_state._replace(
                train_state=train_state,
                buffer_state=buffer_state,
                rng=rng)

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
