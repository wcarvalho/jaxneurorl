"""
Recurrent Q-learning.
"""



import functools
import os
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import NamedTuple, Dict, Union, Optional, Tuple, Callable
from flax.linen.initializers import constant, orthogonal
from flax.typing import Initializer
from flax.linen.linear import default_kernel_init
# from jax.nn.initializers import lecun_normal, orthogonal

import chex

import optax
import flax.linen as nn
import wandb
import matplotlib.pyplot as plt


import flax
from flax import struct
import rlax
from gymnax.environments import environment

from jaxneurorl import losses
from jaxneurorl import loggers
from jaxneurorl.agents.basics import TimeStep
from jaxneurorl.agents import value_based_basics as vbb
from projects.humansf.networks import CategoricalHouzemazeObsEncoder

import pdb

def extract_timestep_input(timestep: TimeStep):
  return RNNInput(
      obs=timestep.observation,
      reset=timestep.first())

Agent = nn.Module
Params = flax.core.FrozenDict
AgentState = flax.struct.PyTreeNode
RNNInput = vbb.RNNInput

def cumulants_from_env(data):
  return data.timestep.observation.state_features  # [T, B, C]

def cumulants_from_preds(
  data,
  online_preds,
  online_state,
  target_preds,
  target_state,
  stop_grad=True,
  use_target=False):

  if use_target:
    cumulants = target_preds.state_feature
  else:
    cumulants = online_preds.state_feature
  if stop_grad:
    return jax.lax.stop_gradient(cumulants) # [T, B, C]
  else:
    return cumulants # [T, B, C]

def episode_mean(x, mask):
  if len(mask.shape) < len(x.shape):
    nx = len(x.shape)
    nd = len(mask.shape)
    extra = nx - nd
    dims = list(range(nd, nd+extra))
    z = jnp.multiply(x, jnp.expand_dims(mask, dims))
  else:
    z = jnp.multiply(x, mask)
  return (z.sum(0))/(mask.sum(0)+1e-5)

def make_episode_mask(data= None, include_final=False, **kwargs):
  """Look at where have valid task data. Everything until 1 before final valid data counts towards task. Data.discount always ends two before final data.
  e.g. if valid data is [x1, x2, x3, 0, 0], data.discount is [1,0,0,0,0]. So can use that to obtain masks.

  NOTE: should probably generalize but have not found need yet.
  Args:
      data (TYPE): Description
      include_final (bool, optional): if True, include all data. if False, include until 1 time-step before final data

  Returns:
      TYPE: Description
  """
  if data.discount.ndim == 2:
    T, B = data.discount.shape
    # for data [x1, x2, x3, 0, 0]
    if include_final:
      # return [1,1,1,0,0]
      return jnp.concatenate((jnp.ones((2, B)), data.discount[:-2]), axis=0)
    else:
      # return [1,1,0,0,0]
      return jnp.concatenate((jnp.ones((1, B)), data.discount[:-1]), axis=0)
  elif data.discount.ndim == 1:
    if include_final:
      return jnp.concatenate((jnp.ones((2,)), data.discount[:-2]), axis=0)
    else:
      return jnp.concatenate((jnp.ones((1,)), data.discount[:-1]), axis=0)
  else:
    raise NotImplementedError

# @dataclasses.dataclass
class UsfaLossFn(vbb.RecurrentLossFn):

  extract_cumulants: Callable = cumulants_from_env
  extract_task: Callable = lambda data: data.observation.observation['task']
  index_cumulants: Callable[
    [jnp.ndarray], jnp.ndarray] = lambda x : x[:-1]
  mask_loss: bool = False

  def error(self, data, online_preds, online_state, target_preds, target_state, steps, **kwargs):
    # ======================================================
    # Prepare Data
    # ======================================================
    # all are [T+1, B, N, A, C]
    # N = num policies, A = actions, C = cumulant dim
    online_sf = online_preds.sf
    # online_z = online_preds.policy
    online_task = online_preds.task
    target_sf = target_preds.sf
    # pseudo rewards, [T/T+1, B, C]
    cumulants = cumulants_from_env(data)
    cumulants = cumulants.astype(online_sf.dtype)

    # Get selector actions from online Q-values for double Q-learning.
    # online_q =  (online_sf*online_z).sum(axis=-1) # [T+1, B, N, A]

    # online_q = jnp.tensordot(online_sf, online_task, axes=([-1], [0]))
    online_q = online_preds.q_vals

    selector_actions = jnp.argmax(online_q, axis=-1) # [T+1, B, N]
    online_actions = data.action # [T, B]

    # Preprocess discounts & rewards.
    discounts = (data.discount * self.discount).astype(online_q.dtype) # [T, B]
    # ======================================================
    # Prepare loss (via vmaps)
    # ======================================================
    td_error_fn = functools.partial(
            rlax.transformed_n_step_q_learning,
            n=20)

    # vmap over batch dimension (B), return B in dim=1
    td_error_fn = jax.vmap(td_error_fn, in_axes=1, out_axes=1)

    # vmap over policy dimension (N), return N in dim=2
    td_error_fn = jax.vmap(td_error_fn, in_axes=(2, None, 2, None, None, None), out_axes=2)

    # vmap over cumulant dimension (C), return in dim=3
    td_error_fn = jax.vmap(td_error_fn, in_axes=(4, None, 4, None, 2, None), out_axes=3)

    # SF loss
    # output = [0=T, 1=B, 2=N, 3=C]
    # pdb.set_trace()
    batch_td_error = td_error_fn(
      online_sf[:-1],  # [T, B, N, A, C] (vmap 2,1)
      online_actions[:-1],  # [T, B]          (vmap None,1)
      target_sf[1:],  # [T, B, N, A, C] (vmap 2,1)
      selector_actions[1:],  # [T, B, N]       (vmap 2,1)
      cumulants[:-1],  # [T, B, C]       (vmap None,1) # reward
      discounts[:-1])  # [T, B]          (vmap None,1)
    # [T, B, N, C] --> [T, B]
    # sum over cumulants, mean over # of policies
    batch_td_error = batch_td_error.sum(axis=3).mean(axis=2)

    if self.mask_loss:
      # [T, B]
      episode_mask = make_episode_mask(data, include_final=False)
      # average over {T, N, C} --> # [B]
      batch_loss = episode_mean(
        x=(0.5 * jnp.square(batch_td_error)),
        mask=episode_mask[:-1])
    else:
      batch_loss = (0.5 * jnp.square(batch_td_error)).mean(axis=(0,)) # 2,3

    metrics = {
      '2.cumulants': cumulants.mean(),
      '2.sf_mean': online_sf.mean(),
      '2.sf_var': online_sf.var(),
      'n_updates': steps
      }
    if self.logger.learner_log_extra is not None:
        self.logger.learner_log_extra({
        'data': data,
        'td_errors': batch_td_error,                 # [T]
        # 'mask': loss_mask,                 # [T]
        'q_values': online_q,    # [T, B]
        'q_loss': batch_loss,                        #[ T, B]
        # 'q_target': target_q_t,
        'cumulants': cumulants.mean(axis=2),
        'online_sf': online_sf.mean(axis=2),
        'n_updates': steps,
    })

    return batch_td_error, batch_loss, metrics # [T, B], [B]

def make_logger(config: dict,
                env: environment.Environment,
                env_params: environment.EnvParams):

    def qlearner_logger(data: dict):
        def callback(d):
            if wandb.run is None: return
            n_updates = d.pop('n_updates')

            # Extract the relevant data
            # only use data from batch dim = 0
            # [T, B, ...] --> # [T, ...]
            d_ = jax.tree_map(lambda x: x[:, 0], d)

            rewards = d_['data'].timestep.reward
            actions = d_['data'].action
            q_values = d_['q_values']
            q_target = d_['q_target']
            q_values_taken = rlax.batched_index(q_values, actions)

            # metrics = {
            #     '2.cumulants': cumulants.mean(),
            #     '2.sf_mean': online_sf.mean(),
            #     '2.sf_var': online_sf.var(),
            #     'n_updates': steps
            # }
            # cumulants = d_['2.cumulants']
            # cumulants_mean = d_['2.sf_mean']

            # Create a figure with three subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))

            def format(ax):
                ax.set_xlabel('Time')
                ax.grid(True)
                ax.set_xticks(range(0, len(rewards), 1))

            # Plot rewards and q-values in the top subplot
            ax1.plot(rewards, label='Rewards')
            ax1.plot(q_values_taken, label='Q-Values')
            ax1.plot(q_target, label='Q-Targets')
            format(ax1)
            ax1.set_title('Rewards and Q-Values')
            ax1.legend()

            # Plot TD errors in the middle subplot
            ax2.plot(d_['td_errors'])
            format(ax2)
            ax2.set_title('TD Errors')

            # Plot Q-loss in the bottom subplot
            ax3.plot(d_['q_loss'])
            format(ax3)
            ax3.set_title('Q-Loss')

            # Adjust the spacing between subplots
            plt.tight_layout()
            # log
            wandb.log({f"learner_details/q-values": wandb.Image(fig)})
            plt.close(fig)


        # this will be the value after update is applied
        n_updates = data['n_updates'] + 1
        is_log_time = n_updates % config["LEARNER_EXTRA_LOG_PERIOD"] == 0

        jax.lax.cond(
            is_log_time,
            lambda d: jax.debug.callback(callback, d),
            lambda d: None,
            data)


    return loggers.Logger(
        gradient_logger=loggers.default_gradient_logger,
        learner_logger=loggers.default_learner_logger,
        experience_logger=loggers.default_experience_logger,
        learner_log_extra=qlearner_logger,
    )

class USFAPreds(NamedTuple):
    q_vals: jnp.ndarray  # q-value
    sf: jnp.ndarray # successor features
    # policy: jnp.ndarray  # policy vector
    task: jnp.ndarray  # task vector (potentially embedded)

class Predictions(NamedTuple):
    q_vals: jax.Array
    rnn_states: jax.Array

class Block(nn.Module):
  features: int
  kernel_init: Initializer = default_kernel_init

  @nn.compact
  def __call__(self, x, _):
    x = nn.Dense(self.features,use_bias=False, kernel_init=self.kernel_init)(x)
    x = jax.nn.relu(x)
    return x, None

class MLP(nn.Module):
  hidden_dim: int
  out_dim: Optional[int] = None
  num_layers: int = 1
  use_bias: bool = False
  kernel_init: Initializer = default_kernel_init

  @nn.compact
  def __call__(self, x):
    for _ in range(self.num_layers):
        x, _ = Block(self.hidden_dim, kernel_init=self.kernel_init)(x, None)

    x = nn.Dense(self.out_dim or self.hidden_dim, kernel_init=self.kernel_init,
                 use_bias=self.use_bias)(x)
    return x

class RnnAgent(nn.Module):
    action_dim: int
    hidden_dim: int
    init_scale: float
    rnn: vbb.ScannedRNN
    use_bias: bool
    observation_encoder: nn.Module
    sf_features: int = 32
    train_tasks: int = 5

    def setup(self):
        self.q_fn = MLP(hidden_dim=self.hidden_dim, num_layers=1, out_dim=self.action_dim, use_bias=self.use_bias)
        # self.state_features = 10 # SF feature dimensions
        self.sf_nets = [MLP(
           hidden_dim=self.hidden_dim, num_layers=1, use_bias=self.use_bias, out_dim=self.action_dim * self.sf_features,
           kernel_init=orthogonal(2), # From craftax
           ) for task in range(self.train_tasks)]

    def initialize(self, x: TimeStep):
        """Only used for initialization."""
        # [B, D]
        rng = jax.random.PRNGKey(0)
        batch_dims = (x.reward.shape[0],)
        rnn_state = self.initialize_carry(rng, batch_dims)

        return self.__call__(rnn_state, x, rng)

    def initialize_carry(self, *args, **kwargs):
        """Initializes the RNN state."""
        return self.rnn.initialize_carry(*args, **kwargs)

    def __call__(self, rnn_state, x: TimeStep, rng: jax.random.PRNGKey):
        x = extract_timestep_input(x)

        embedding = self.observation_encoder(x.obs)
        embedding = nn.relu(embedding)

        rnn_in = x._replace(obs=embedding)
        rng, _rng = jax.random.split(rng)
        new_rnn_state, rnn_out = self.rnn(rnn_state, rnn_in, _rng)

        task_w = x.obs.task_w

        return self.sfgpi(rnn_out, task_w, x.obs.train_vector), new_rnn_state

    def unroll(self, rnn_state, xs: TimeStep, rng: jax.random.PRNGKey):
        # rnn_state: [B]
        # xs: [T, B]
        xs = extract_timestep_input(xs)

        embedding = jax.vmap(self.observation_encoder, in_axes=0)(xs.obs)
        embedding = nn.relu(embedding)

        rnn_in = xs._replace(obs=embedding)
        rng, _rng = jax.random.split(rng)
        new_rnn_state, rnn_out = self.rnn.unroll(rnn_state, rnn_in, _rng)

        task_w = xs.obs.task_w

        return jax.vmap(self.sfgpi, in_axes=0)(rnn_out, task_w, xs.obs.train_vector), new_rnn_state

    def sfgpi(self,
        usfa_input: jnp.ndarray,
        task: jnp.ndarray,
        train_vector: jnp.ndarray
        ) -> USFAPreds:
        """Summary

        Args:
            usfa_input (jnp.ndarray): D, typically rnn_output
            policies (jnp.ndarray): N x D
            task (jnp.ndarray): D
        Returns:
            USFAPreds: Description
        """
        def compute_sf_q(sf_input: jnp.ndarray,
                       task: jnp.ndarray,
                       train_vector) -> jnp.ndarray:
            """Compute successor features and q-valuesu

            Args:
                inputs (jnp.ndarray): D_1
                policy (jnp.ndarray): D_2
                task (jnp.ndarray): D_1

            Returns:
                jnp.ndarray: 2-D tensor of action values of shape [batch_size, num_actions]
            """
            # TODO: should there be a policy? or how to we iterate over train tasks?
            # sf_input = jnp.concatenate((sf_input, policy))  # 2D

            # [A * C]
            # [N, A * C]
            # When not in evaluation, the iteration should be over the current task only, not gpi.
            tasks = self.train_tasks
            sf = jnp.array([self.sf_nets[idx](sf_input) for idx in range(tasks)])
            # Multiply the sfs with train vector. If train task, only particular task taken, if test task, all SFs valid.
            sf = jax.vmap(lambda a,b: a*b, in_axes=(0, 0))(sf, train_vector)
            # [N, A, C]
            sf = jnp.reshape(sf, (tasks, self.action_dim, self.sf_features))

            # [B, N, A]
            # train_task = self.tasks
            # eval_task = obs.task
            # train_tasks is one of the outputs of the environment.
            # Test tasks are some LC of the train tasks.
            # Set env_params for test.
            # New reset function - sample a task - train or test
            result = jnp.tensordot(sf, task, axes=([-1], [0]))
            # [B, A]
            q_values = jnp.max(result, axis=-2)

            assert q_values.shape[0] == self.action_dim, 'wrong shape'
            return sf, q_values

        # policy_embeddings = self.policy_net(policies)
        sfs, q_values = jax.vmap(
        compute_sf_q, in_axes=(0, 0, 0), out_axes=0)(
            usfa_input,
            task,
            train_vector)

        # GPI
        # -----------------------
        # [N, A] --> [A]
        # q_values = jnp.max(q_values, axis=-2)

        # policies = expand_tile_dim(policies, axis=-2, size=self.num_actions)

        return USFAPreds(
            sf=sfs,       # [N, A, D_w]
            # policy=policies,         # [N, A, D_w]
            q_vals=q_values,  # [N, A]
            task=task)         # [D_w]

def epsilon_greedy_act(q, eps, key):
    key_a, key_e   = jax.random.split(key, 2) # a key for sampling random actions and one for picking
    greedy_actions = jnp.argmax(q, axis=-1) # get the greedy actions TODO: update this to take action by SF.
    random_actions = jax.random.randint(key_a, shape=greedy_actions.shape, minval=0, maxval=q.shape[-1]) # sample random actions
    pick_random    = jax.random.uniform(key_e, greedy_actions.shape)<eps # pick which actions should be random
    chosen_actions = jnp.where(pick_random, random_actions, greedy_actions)
    return chosen_actions

class LinearDecayEpsilonGreedy:
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

        eps = self.get_epsilon(t)
        rng = jax.random.split(rng, q_vals.shape[0])
        return jax.vmap(epsilon_greedy_act, in_axes=(0, None, 0))(q_vals, eps, rng)

class FixedEpsilonGreedy:
    """Epsilon Greedy action selection"""

    def __init__(self, epsilons: float):
        self.epsilons = epsilons

    @partial(jax.jit, static_argnums=0)
    def choose_actions(self, q_vals: jnp.ndarray, t: int, rng: chex.PRNGKey):

        rng = jax.random.split(rng, q_vals.shape[0])
        return jax.vmap(epsilon_greedy_act, in_axes=(0, 0, 0))(q_vals, self.epsilons, rng)

def make_rnn_agent(
        config: dict,
        env: environment.Environment,
        env_params: environment.EnvParams,
        example_timestep: TimeStep,
        rng: jax.random.PRNGKey) -> Tuple[nn.Module, Params, vbb.AgentResetFn]:

    agent = RnnAgent(
        action_dim=env.num_actions() + 1,
        hidden_dim=config["AGENT_HIDDEN_DIM"],
        init_scale=config['AGENT_INIT_SCALE'],
        use_bias=config['AGENT_USE_BIAS'],
        rnn=vbb.ScannedRNN(hidden_dim=config["AGENT_HIDDEN_DIM"]),
        observation_encoder=CategoricalHouzemazeObsEncoder(env.num_categories),
        sf_features=len(env_params.objects),
        train_tasks=env_params.n_train
    )

    rng, _rng = jax.random.split(rng)
    network_params = agent.init(
        _rng, example_timestep,
        method=agent.initialize)

    def reset_fn(params, example_timestep, reset_rng):
      batch_dims = (example_timestep.reward.shape[0],)
      return agent.apply(
          params,
          batch_dims=batch_dims,
          rng=reset_rng,
          method=agent.initialize_carry)

    return agent, network_params, reset_fn

def make_mlp_agent(
        config: dict,
        env: environment.Environment,
        env_params: environment.EnvParams,
        example_timestep: TimeStep,
        rng: jax.random.PRNGKey) -> Tuple[nn.Module, Params, vbb.AgentResetFn]:

    agent = RnnAgent(
        action_dim=env.action_space(env_params).n,
        hidden_dim=config["AGENT_HIDDEN_DIM"],
        init_scale=config['AGENT_INIT_SCALE'],
        rnn=vbb.DummyRNN()
    )

    rng, _rng = jax.random.split(rng)
    network_params = agent.init(
        _rng, example_timestep,
        method=agent.initialize)

    def reset_fn(params, example_timestep, reset_rng):
      del params
      del reset_rng
      batch_dims = example_timestep.observation.shape[:-1]
      return jnp.zeros(batch_dims)

    return agent, network_params, reset_fn

def make_optimizer(config: dict) -> optax.GradientTransformation:
  def linear_schedule(count):
      frac = 1.0 - (count / config["NUM_UPDATES"])
      return config["LR"] * frac

  lr = linear_schedule if config.get("LR_LINEAR_DECAY", False) else config["LR"]

  return optax.chain(
      optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
      optax.adam(learning_rate=lr, eps=config['EPS_ADAM'])
  )

def make_loss_fn_class(config) -> vbb.RecurrentLossFn:
  return functools.partial(
     # R2D2LossFn,
     UsfaLossFn,
     discount=config['GAMMA'])

def make_actor(config: dict, agent: Agent, rng: jax.random.PRNGKey) -> vbb.Actor:
    fixed_epsilon = config.get('FIXED_EPSILON', 1)
    assert fixed_epsilon in (0, 1, 2)
    if fixed_epsilon:
        ## BELOW was copied from ACME
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
        train_state: vbb.TrainState,
        agent_state: jax.Array,
        timestep: TimeStep,
        rng: jax.random.PRNGKey):
        preds, agent_state = agent.apply(
            train_state.params, agent_state, timestep, rng)

        action = explorer.choose_actions(
            preds.q_vals, train_state.timesteps, rng)

        return preds, action, agent_state

    def eval_step(
        train_state: vbb.TrainState,
        agent_state: jax.Array,
        timestep: TimeStep,
        rng: jax.random.PRNGKey):
        preds, agent_state = agent.apply(
            train_state.params, agent_state, timestep, rng)

        # Use a GPI in this function, can use GPI for eval step
        action = preds.q_vals.argmax(-1)

        return preds, action, agent_state

    return vbb.Actor(train_step=actor_step, eval_step=eval_step)

make_train_preloaded = functools.partial(
   vbb.make_train,
   make_agent=make_rnn_agent,
   make_optimizer=make_optimizer,
   make_loss_fn_class=make_loss_fn_class,
   make_actor=make_actor,
   make_logger=make_logger,
)