"""
Universal Successor Feature Approximator (USFA)
"""

import functools
from typing import NamedTuple, Dict, Union, Optional, Tuple, Callable

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

import chex
import optax
import flax.linen as nn
import flax
from flax import struct
import rlax
from gymnax.environments import environment

from jaxneurorl import losses
from jaxneurorl import loggers
from jaxneurorl.agents.basics import TimeStep
from jaxneurorl.agents import value_based_basics as vbb

Params = flax.core.FrozenDict

@struct.dataclass
class UsfaR2D2LossFn(vbb.RecurrentLossFn):
    extract_cumulants: Callable = lambda data: data.timestep.observation.state_features
    extract_task: Callable = lambda data: data.timestep.observation.task_w

    def error(self, data, online_preds, online_state, target_preds, target_state, steps, **kwargs):
        # Prepare Data
        online_sf = online_preds.sf  # [T+1, B, N, A, C]
        online_w = data.timestep.observation.task_w  # [T+1, B, C]
        target_sf = target_preds.sf  # [T+1, B, N, A, C]

        cumulants = self.extract_cumulants(data=data)
        cumulants = cumulants.astype(online_sf.dtype)

        # Get selector actions from online Q-values for double Q-learning
        dot = lambda x, y: (x * y).sum(axis=-1)
        vdot = jax.vmap(jax.vmap(dot, (2, None), 2), (2, None), 2)
        online_q = vdot(online_sf, online_w)  # [T+1, B, N, A]
        selector_actions = jnp.argmax(online_q, axis=-1)  # [T+1, B, N]

        # Preprocess discounts & rewards
        def float(x): return x.astype(jnp.float32)
        discounts = float(data.discount) * self.discount
        lambda_ = jnp.ones_like(data.discount) * self.lambda_
        is_last = float(data.is_last)

        # Prepare loss (via vmaps)
        # vmap over batch dimension (B), return B in dim=1
        td_error_fn = jax.vmap(
            losses.q_learning_lambda_td,
            in_axes=1, out_axes=1
        )
        # vmap over policy dimension (N), return N in dim=2
        td_error_fn = jax.vmap(td_error_fn, in_axes=(
            2, None, 2, 2, None, None, None, None), out_axes=2)

        # vmap over cumulant dimension (C), return in dim=3
        td_error_fn = jax.vmap(td_error_fn, in_axes=(
            4, None, 4, None, 2, None, None, None), out_axes=3)

        # [T, B, N, C]
        sf_t, target_sf_t = td_error_fn(
            online_sf[:-1],  # [T, B, N, A, C]
            data.action[:-1],  # [T, B]
            target_sf[1:],  # [T, B, N, A, C]
            selector_actions[1:],  # [T, B, N]
            cumulants[1:],  # [T, B, C]
            discounts[1:],
            is_last[1:],
            lambda_[1:]
        )

        # Ensure target = 0 when episode terminates
        target_sf_t = target_sf_t * data.discount[:-1, :, None, None]
        batch_td_error = target_sf_t - sf_t

        # Ensure loss = 0 when episode truncates
        truncated = (data.discount + is_last) > 1
        loss_mask = (1 - truncated).astype(batch_td_error.dtype)[:-1, :, None, None]
        batch_td_error = batch_td_error * loss_mask

        # [T, B, N, C]
        batch_loss = 0.5 * jnp.square(batch_td_error)

        # [B]
        batch_loss_mean = (batch_loss * loss_mask).mean(axis=(0, 2, 3))

        metrics = {
            '0.sf_loss': batch_loss.mean(),
            '0.sf_td': jnp.abs(batch_td_error).mean(),
            '1.cumulants': cumulants.mean(),
            'z.sf_mean': online_sf.mean(),
            'z.sf_var': online_sf.var(),
        }

        if self.logger.learner_log_extra is not None:
            self.logger.learner_log_extra({
                'data': data,
                'cumulants': cumulants[1:],
                'td_errors': jnp.abs(batch_td_error),  # [T, B, N, C]
                'mask': loss_mask[:, :, 0, 0],  # [T, B]
                'sf_values': online_sf[:-1],  # [T+1, B, N, A, C]
                'sf_loss': batch_loss,  # [T, B, N, C]
                'sf_target': target_sf_t,
                'n_updates': steps,
            })

        return batch_td_error, batch_loss_mean, metrics  # [T, B, N, C], [B]


def make_optimizer(config: dict) -> optax.GradientTransformation:
  def linear_schedule(count):
      frac = 1.0 - (count / config["NUM_UPDATES"])
      return config["LR"] * frac

  lr = linear_schedule if config.get(
      "LR_LINEAR_DECAY", False) else config["LR"]

  return optax.chain(
      optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
      optax.adam(learning_rate=lr, eps=config['EPS_ADAM'])
  )


def make_loss_fn_class(config) -> vbb.RecurrentLossFn:
  return functools.partial(
      UsfaR2D2LossFn,
      discount=config['GAMMA'],
      )







class USFAPreds(NamedTuple):
    q_vals: jnp.ndarray  # q-value
    sf: jnp.ndarray  # successor features
    policy: jnp.ndarray  # policy vector
    task: jnp.ndarray  # task vector (potentially embedded)

def extract_timestep_input(timestep: TimeStep):
    return vbb.RNNInput(
        obs=timestep.observation,
        reset=timestep.first())

def sample_gauss(mean, var, key, nsamples):
    import ipdb; ipdb.set_trace()
    if nsamples >= 1:
        mean = jnp.expand_dims(mean, -2)  # [1, ]
        samples = jnp.tile(mean, [1, nsamples, 1])
        dims = samples.shape  # [N, D]
        samples = samples + jnp.sqrt(var) * jax.random.normal(key, dims)
        samples = samples.astype(mean.dtype)
    else:
        samples = jnp.expand_dims(mean, axis=-1)  # [N, D]
    return samples


def get_task_onehot(task_vector, train_tasks):
    # Compare the task_vector with each train_task
    matches = jnp.all(task_vector == train_tasks, axis=1)
    # Create a one-hot vector based on the match
    one_hot = jnp.eye(len(train_tasks))[jnp.argmax(matches)]
    return one_hot

class MLP(nn.Module):
  hidden_dim: int
  out_dim: Optional[int] = None
  num_layers: int = 1

  @nn.compact
  def __call__(self, x):
    for _ in range(self.num_layers):
        x = nn.Dense(self.hidden_dim, use_bias=False)(x)
        x = jax.nn.relu(x)

    x = nn.Dense(self.out_dim or self.hidden_dim, use_bias=False)(x)
    return x

class SfGpiHead(nn.Module):
    num_actions: int
    state_features_dim: int
    train_tasks: jnp.ndarray
    num_layers: int = 2
    hidden_dim: int = 512
    nsamples: int = 10
    variance: float = 0.5
    eval_task_support: str = 'train'

    def setup(self):
        self.policy_net = lambda x: x
        self.sf_net = MLP(
           hidden_dim=self.hidden_dim,
           num_layers=self.num_layers,
           out_dim=self.num_actions * self.state_features_dim)

    @nn.compact
    def __call__(self, usfa_input: jnp.ndarray, task: jnp.ndarray) -> USFAPreds:
        policy = get_task_onehot(task, self.train_tasks)
        if self.nsamples > 1:
            policy_samples = sample_gauss(
                mean=policy,
                var=self.variance,
                key=self.make_rng('sample'),
                nsamples=self.nsamples-1, axis=-2)
            policy_base = jnp.expand_dims(policy, axis=-2)  # [1, D_w]
            policies = jnp.concatenate((policy_base, policy_samples), axis=-2)  # [N+1, D_w]
        else:
            policies = jnp.expand_dims(policy, axis=-2)  # [1, D_w]

        return self.sfgpi(usfa_input=usfa_input, policies=policies, task=task)

    def evaluate(self,
                 usfa_input: jnp.ndarray,
                 task: jnp.ndarray) -> USFAPreds:

        if self.eval_task_support == 'train':
            policies = jnp.array([get_task_onehot(task, self.train_tasks) for task in self.train_tasks])
        elif self.eval_task_support == 'eval':
            policies = jnp.expand_dims(
                get_task_onehot(task, self.train_tasks), axis=-2)
        elif self.eval_task_support == 'train_eval':
            task_expand = jnp.expand_dims(
                get_task_onehot(task, self.train_tasks), axis=-2)
            train_policies = jnp.array([get_task_onehot(task, self.train_tasks) for task in self.train_tasks])
            policies = jnp.concatenate((train_policies, task_expand), axis=-2)
        else:
            raise RuntimeError(self.eval_task_support)

        return self.sfgpi(usfa_input=usfa_input, policies=policies, task=task)

    def sfgpi(self, usfa_input: jnp.ndarray, policies: jnp.ndarray, task: jnp.ndarray) -> USFAPreds:
        def compute_sf_q(sf_input: jnp.ndarray, policy: jnp.ndarray, task: jnp.ndarray):
            sf_input = jnp.concatenate((sf_input, policy))  # 2D
            sf = self.sf_net(sf_input)
            sf = jnp.reshape(sf, (self.num_actions, self.state_features_dim))
            q_values = jnp.sum(sf * task, axis=-1)
            return sf, q_values

        # []
        policy_embeddings = self.policy_net(policies)
        sfs, q_values = jax.vmap(compute_sf_q, in_axes=(None, 0, None), out_axes=0)(
            usfa_input, policy_embeddings, task)

        q_values = jnp.max(q_values, axis=-2)
        policies = jnp.expand_dims(policies, axis=-2)
        policies = jnp.tile(policies, (1, self.num_actions, 1))

        return USFAPreds(sf=sfs, policy=policies, q_vals=q_values, task=task)

class UsfaAgent(nn.Module):
    observation_encoder: nn.Module
    rnn: vbb.ScannedRNN
    sf_head: SfGpiHead

    def initialize(self, x: TimeStep):
        rng = jax.random.PRNGKey(0)
        batch_dims = (x.reward.shape[0],)
        rnn_state = self.initialize_carry(rng, batch_dims)
        return self.__call__(rnn_state, x, rng)

    def initialize_carry(self, *args, **kwargs):
        return self.rnn.initialize_carry(*args, **kwargs)

    def __call__(self, rnn_state, x: TimeStep, rng: jax.random.PRNGKey, evaluate: bool = False):

        embedding = self.observation_encoder(x.observation)
        rnn_in = vbb.RNNInput(obs=embedding, reset=x.first())
        rng, _rng = jax.random.split(rng)
        new_rnn_state, rnn_out = self.rnn(rnn_state, rnn_in, _rng)
        
        if evaluate:
            predictions = jax.vmap(self.sf_head.evaluate)(
              rnn_out, x.observation.task_w)
        else:
            predictions = jax.vmap(self.sf_head)(
              # [B, D], [B, D]
              rnn_out, x.observation.task_w)
        
        return predictions, new_rnn_state

    def unroll(self, rnn_state, xs: TimeStep, rng: jax.random.PRNGKey):

        embedding = nn.BatchApply(self.observation_encoder)(xs.observation)
        rnn_in = vbb.RNNInput(obs=embedding, reset=xs.first())
        rng, _rng = jax.random.split(rng)
        new_rnn_state, rnn_out = self.rnn.unroll(rnn_state, rnn_in, _rng)

        predictions = jax.vmap(jax.vmap(self.sf_head))(rnn_out, xs.observation.task_w)

        return predictions, new_rnn_state

def make_agent(
        config: dict,
        env: environment.Environment,
        env_params: environment.EnvParams,
        example_timestep: TimeStep,
        rng: jax.random.PRNGKey,
        train_tasks: jnp.ndarray,
        ObsEncoderCls: nn.Module) -> Tuple[nn.Module, Params, vbb.AgentResetFn]:

    sf_head = SfGpiHead(
        num_actions=env.num_actions(env_params),
        state_features_dim=example_timestep.observation.task_w.shape[-1],
        nsamples=config.get('NSAMPLES', 1),
        eval_task_support=config.get('EVAL_TASK_SUPPORT', 'train'),
        train_tasks=train_tasks,
        num_layers=config.get('NUM_SF_LAYERS', 2),
        hidden_dim=config.get('SF_HIDDEN_DIM', 512),
    )

    agent = UsfaAgent(
        observation_encoder=ObsEncoderCls(),
        rnn=vbb.ScannedRNN(hidden_dim=config["AGENT_RNN_DIM"]),
        sf_head=sf_head
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

class FixedEpsilonGreedy:
    """Epsilon Greedy action selection"""

    def __init__(self, epsilons: float):
        self.epsilons = epsilons

    @partial(jax.jit, static_argnums=0)
    def choose_actions(self, q_vals: jnp.ndarray, t: int, rng: chex.PRNGKey):

        rng = jax.random.split(rng, q_vals.shape[0])
        return jax.vmap(epsilon_greedy_act, in_axes=(0, 0, 0))(q_vals, self.epsilons, rng)

def make_actor(
        config: dict,
        agent: nn.Module,
        rng: jax.random.PRNGKey,
        remove_gpi_dim: bool = True,
        ) -> vbb.Actor:
    fixed_epsilon = config.get('FIXED_EPSILON', 1)
    assert fixed_epsilon in (0, 1, 2)
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

    def actor_step(
            train_state: vbb.TrainState,
            agent_state: jax.Array,
            timestep: TimeStep,
            rng: jax.random.PRNGKey):
        preds, agent_state = agent.apply(
            train_state.params, agent_state, timestep, rng)

        action = explorer.choose_actions(
            preds.q_vals, train_state.timesteps, rng)

        if remove_gpi_dim:
            preds = preds._replace(sf=preds.sf[:, 0], policy=preds.policy[:, 0])

        return preds, action, agent_state

    def eval_step(
            train_state: vbb.TrainState,
            agent_state: jax.Array,
            timestep: TimeStep,
            rng: jax.random.PRNGKey):
        preds, agent_state = agent.apply(
            train_state.params, agent_state, timestep, rng, evaluate=True)
        action = preds.q_vals.argmax(-1)

        if remove_gpi_dim:
            preds = preds._replace(sf=preds.sf[:, 0], policy=preds.policy[:, 0])

        return preds, action, agent_state

    return vbb.Actor(train_step=actor_step, eval_step=eval_step)

def make_train(
    make_agent=make_agent,
    make_optimizer=make_optimizer,
    make_loss_fn_class=make_loss_fn_class,
    make_actor=make_actor,
    **kwargs
):
    return vbb.make_train(
        make_agent=make_agent,
        make_optimizer=make_optimizer,
        make_loss_fn_class=make_loss_fn_class,
        make_actor=make_actor,
        **kwargs,
    )