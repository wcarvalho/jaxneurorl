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

import distrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mctx
import optax
import rlax
import wandb

from library import utils
from library import loggers
from library import losses

from agents.basics import TimeStep
from agents import value_based_basics as vbb

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
    policy_target: Optional[mctx.PolicyOutput] = None


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
    discretizer: utils.Discretizer = None
    lambda_: float = .9
    policy_coef: float = 1.0
    value_coef: float = 0.25

    def loss_fn(self,
                data,
                online_preds,
                target_preds,
                batch_index,
                steps: int):
        """This will compute the loss for the policy and value function.

        For the policy, either (a) use the policy logits from the experience or (b) generate new ones using MCTS.

        For value, either use the environment return or (b) generate new ones from MCTS.

        For simplicity, we will not re-analyze.
        TODO: implement reanalyze.
        """
        is_last = data.is_last
        ################
        # Policy loss
        ################
        # ---------------
        # target
        # ---------------
        # [T, A]
        preds = data.extras.get('preds')
        policy_target = preds.policy_target
        # [T] --> [T, A]
        random_policy_mask = jnp.broadcast_to(
            is_last[:, None], policy_target.shape)
        num_actions = online_preds.policy_logits.shape[-1]
        uniform_policy = jnp.ones_like(policy_target) / num_actions

        policy_probs_target = jax.lax.select(
            random_policy_mask, uniform_policy, policy_target)

        # ---------------
        # loss
        # ---------------
        policy_ce = jax.vmap(rlax.categorical_cross_entropy)(
            policy_probs_target, online_preds.policy_logits)

        # []
        policy_loss = self.policy_coef*policy_ce.mean()

        ################
        # Value loss
        ################
        target_net_values = self.discretizer.logits_to_scalar(
            target_preds.value_logits)
        lambda_ = jnp.ones_like(data.discount)*self.lambda_

        lambda_ *= (1 - is_last.astype(lambda_.dtype))
        value_target = rlax.lambda_returns(
            data.reward[1:],
            data.discount[1:]*self.discount,
            target_net_values[1:],
            lambda_[1:],
        )
        #value_target = losses.n_step_target(
        #    target_net_values[1:],
        #    data.reward[1:],
        #    data.discount[1:]*self.discount,
        #    is_last[1:],
        #    lambda_[1:],
        #)
        value_target = value_target*data.discount[:-1]
        value_probs_target = self.discretizer.scalar_to_probs(value_target)

        num_v_preds = value_probs_target.shape[0]
        value_ce = jax.vmap(rlax.categorical_cross_entropy)(
            value_probs_target,
            online_preds.value_logits[:num_v_preds])

        # []
        # truncated is discount on AND is last
        truncated = (data.discount+is_last) > 1
        loss_mask = (1-truncated).astype(value_ce.dtype)
        value_ce = value_ce*loss_mask[:num_v_preds]
        value_loss = self.value_coef*value_ce.mean()
        total_loss = policy_loss + value_loss

        # ---------
        # TD-error
        # ---------
        value_prediction = self.discretizer.logits_to_scalar(
            online_preds.value_logits[:num_v_preds])
        # [T]
        td_error = value_prediction - value_target

        ################
        # metrics
        ################
        
        metrics = {
            "0.0.total_loss": total_loss,
            "0.0.td-error": jnp.abs(td_error),
            '0.1.policy_loss': policy_loss,
            '0.2.value_loss': value_loss,
        }

        if self.logger.learner_log_extra is not None:
            jax.lax.cond(
                batch_index < 1,
                lambda: self.logger.learner_log_extra({
                    'batch_index': batch_index,
                    'data': data,
                    'td_errors': td_error,       # [T]
                    'mask': loss_mask,           # [T]
                    'values': value_prediction,  # [T]
                    'values_targets': value_target,  # [T]
                    'value_loss': value_ce,    # [T]
                    'policy_loss': policy_ce,    # [T]
                    'n_updates': steps,
                }),
                lambda: None
            )

        return td_error, total_loss, metrics


    def error(self, data, online_preds, online_state, target_preds, target_state, steps, **kwargs):
        assert self.discretizer is not None, 'please set'

        # [B, T], [B], [B, T]
        batch_indices = jnp.arange(data.reward.shape[1])
        td_error, total_loss, metrics = jax.vmap(
           self.loss_fn, in_axes=(1,1,1,0,None), out_axes=0)(
            data,          # [T, B, ...]
            online_preds,  # [T, B, ...]
            target_preds,  # [T, B, ...]
            batch_indices,  # [B]
            steps,
        )
        td_error = td_error.transpose()  # [B,T] --> [T,B]

        # [T, B], [B], []
        return td_error, total_loss, metrics


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

        rnn_in = vbb.RNNInput(obs=embedding, reset=xs.first())
        rng, _rng = jax.random.split(rng)
        new_rnn_state, new_rnn_states = self.rnn.unroll(rnn_state, rnn_in, _rng)

        rnn_out = self.rnn.output_from_state(new_rnn_states)
        policy_logits = nn.BatchApply(self.policy_fn)(rnn_out)
        value_logits = nn.BatchApply(self.value_fn)(rnn_out)
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
      eval_mcts_policy: Optional[mctx.gumbel_muzero_policy] = None,
      ) -> vbb.Actor:
    del rng
    eval_mcts_policy = eval_mcts_policy or mcts_policy

    def actor_step(
            train_state: vbb.TrainState,
            agent_state: jax.Array,
            timestep: TimeStep,
            rng: jax.random.KeyArray,
            evaluation: bool = False,
            ):
        """
        
        Note: some weird things. For some reason, I can't vmap
          over the environment when it's being called within MCTS.
          FIX: vmap over MCTS with each individual call applying the 
            model (which uses ground-truth env) on a per state-action 
            pair basis. In summary:
            Before:
                - MCTS --> Apply Model --> VMAP(ENV)(state, action)
            Now:
                - VMAP(MCTS) --> Apply Model --> ENV(state, action)
            
        """
        preds, agent_state = agent.apply(
            train_state.params, agent_state, timestep, rng)

        value = discretizer.logits_to_scalar(preds.value_logits)

        root = mctx.RootFnOutput(
           prior_logits=preds.policy_logits,
           value=value,
           embedding=preds.state)

        # will vmap over mcts
        # mcts excepts []
        # [B,...] --> [B, 1, ...]
        root = jax.tree_map(lambda x: x[:, None], root)
        rng, improve_key = jax.random.split(rng)

        def apply_mcts_policy(root_, discount_):
            # 1 step of policy improvement
            policy = eval_mcts_policy if evaluation else mcts_policy
            return policy(
                params=train_state.params,
                rng_key=improve_key,
                root=root_,
                recurrent_fn=functools.partial(
                    model_step,
                    discount=discount_,
                    agent=agent,
                    discretizer=discretizer,
                    evaluation=evaluation,
                ))
        mcts_outputs = jax.vmap(apply_mcts_policy)(
            root, timestep.discount[:, None])

        # [B, 1, ...] --> [B, ...]
        mcts_outputs = jax.tree_map(lambda x: x[:, 0], mcts_outputs)

        policy_target = mcts_outputs.action_weights
        if evaluation:
            if config.get('GREEDY_EVAL', True):
                action = jnp.argmax(policy_target, axis=-1)
            else:
                rng, rng_ = jax.random.split(rng)
                action = distrax.Categorical(probs=policy_target).sample(seed=rng_)
        else:
            rng, rng_ = jax.random.split(rng)
            action = distrax.Categorical(probs=policy_target).sample(seed=rng_)

        preds = preds.replace(policy_target=policy_target)

        return preds, action, agent_state

    return vbb.Actor(train_step=actor_step,
                     eval_step=functools.partial(
                         actor_step, evaluation=True))


def learner_log_extra(
        data: dict,
        config: dict,
    ):
    """Note: currently not using. I think running this inside vmap leads to a race conditon where matplotlib stalls indefinitely."""
    def callback(d):
        rewards = d['data'].timestep.reward
        values = d['values']
        values_target = d['values_targets']

        # Create a figure with three subplots
        nplots = 4
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(
            nplots, 1, figsize=(5, 3*nplots))

        # Plot rewards and q-values in the top subplot
        def format(ax):
            ax.set_xlabel('Time')
            ax.grid(True)
            ax.set_xticks(range(0, len(rewards), 1))

        # Plot rewards and q-values in the top subplot
        ax1.plot(rewards, label='Rewards')
        ax1.plot(values, label='Value Predictions')
        ax1.plot(values_target, label='Value Targets')
        format(ax1)
        ax1.set_title('Rewards and Values')
        ax1.legend()

        # Plot TD errors in the middle subplot
        ax2.plot(d['td_errors'])
        format(ax2)
        ax2.set_title('TD Errors')

        # Plot Value-loss in the bottom subplot
        ax3.plot(d['value_loss'])
        format(ax3)
        ax3.set_title('Value Loss')

        # Plot Value-loss in the bottom subplot
        ax4.plot(d['policy_loss'])
        format(ax4)
        ax4.set_title('Policy Loss')

        # Adjust the spacing between subplots
        plt.tight_layout()
        # log
        if wandb.run is not None:
            wandb.log({f"learner_details/losses": wandb.Image(fig)})
        plt.close(fig)

    if config["LEARNER_EXTRA_LOG_PERIOD"] < 1:
        return
    # this will be the value after update is applied
    n_updates = data['n_updates'] + 1
    is_log_time = n_updates % config["LEARNER_EXTRA_LOG_PERIOD"] == 0
    is_log_time = jnp.logical_and(is_log_time, data['batch_index'] < 1)
    jax.lax.cond(
        is_log_time,
        lambda d: jax.debug.callback(callback, d),
        lambda d: None,
        data)


def make_train_preloaded(config, test_env_params=None):
    max_value = config.get('MAX_VALUE', 10)
    num_bins = config['NUM_BINS']

    discretizer = utils.Discretizer(
        max_value=max_value,
        num_bins=num_bins,
        min_value=-max_value)

    mcts_policy = functools.partial(
        mctx.gumbel_muzero_policy,
        max_depth=config.get('MAX_SIM_DEPTH', None),
        num_simulations=config.get('NUM_SIMULATIONS', 2),
        gumbel_scale=config.get('GUMBEL_SCALE', 1.0))

    def make_logger(config: dict,
                    env: environment.Environment,
                    env_params: environment.EnvParams):

        return loggers.Logger(
            gradient_logger=loggers.default_gradient_logger,
            learner_logger=loggers.default_learner_logger,
            experience_logger=loggers.default_experience_logger,
            learner_log_extra=functools.partial(
                learner_log_extra, config=config))

    return functools.partial(
        vbb.make_train,
        make_agent=functools.partial(
            make_agent,
            test_env_params=test_env_params),
        make_optimizer=make_optimizer,
        make_loss_fn_class=functools.partial(
            make_loss_fn_class,
            discretizer=discretizer),
        make_actor=functools.partial(
            make_actor,
            discretizer=discretizer,
            mcts_policy=mcts_policy),
        make_logger=make_logger,
    )
