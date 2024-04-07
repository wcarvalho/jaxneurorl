import functools
import os
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import NamedTuple, Dict, Union, Optional, Tuple, Callable, Any, Sequence
from acme.jax.networks import duelling #? not sure

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
from gymnax.environments import environment


from library.wrappers import TimeStep
from singleagent import value_based_basics as vbb

def extract_timestep_input(timestep: TimeStep):
  return RNNInput(
      obs=timestep.observation,
      done=timestep.last())

Agent = nn.Module # base class for all NN layers & models in Flax
Params = flax.core.FrozenDict # immutable dict
AgentState = flax.struct.PyTreeNode
RNNInput = vbb.RNNInput #? not sure if we still need this?

# PARAMS:
    # tx_pair: Transformation pair used in loss computation.
    # extract_q: Function to extract Q-values from predictions.
    # bootstrap_n: Number of steps for bootstrapping in the loss calculation.
# OUTPUT:
    # computes R2D2 loss, returning batch TD error, batch loss,
    # and a dictionary of metrics
@dataclasses.dataclass
class R2D2LossFn(vbb.RecurrentLossFn):

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

    mask = data.discount[:-1]  # if 0, episode ending
    batch_loss = 0.5 * jnp.square(batch_td_error).mean(axis=0)

    batch_loss = vbb.maked_mean(batch_loss, mask)

    metrics = {
        '0.q_loss': batch_loss.mean(),
        '0.q_td': jnp.abs(batch_td_error).mean(),
        '1.reward': rewards.mean(),
        'z.q_mean': self.extract_q(online_preds).mean(),
        'z.q_var': self.extract_q(online_preds).var(),
        }

    return batch_td_error, batch_loss, metrics  # [T-1, B], [B]

#!
# class Predictions(NamedTuple):
#     q_vals: jax.Array
#     rnn_states: jax.Array
@chex.dataclass(frozen=True)
class Predictions:
  state: jax.Array
  q_values: jax.Array
  rewards: jax.Array

#! modified the following based on DummyRNN in networks.py in old code base
class DummyRNN(nn.Module):
    @nn.compact
    def __call__(self, inputs: jax.Array, prev_state: Any
                ) -> Tuple[jax.Array, Tuple[jax.Array, Any]]:
        #? in old codebase we were using LSTMState, but I am not sure if we want that here
        # so i use Any as a placeholder here
        # Simply returns the inputs and previous state unchanged
        return inputs, prev_state

    def initial_state(self, batch_size: Optional[int] = None) -> Any:
        # Generates an initial state for the RNN, which is a tuple of two zero arrays.
        state_shape = (batch_size, 1) if batch_size is not None else (1,)
        return jnp.zeros(state_shape)

#! modified the SimpleTransition from old codebase (written in haiku)
# need to use this for the transition function
class SimpleTransition(nn.Module):
    num_blocks: int

    @nn.compact
    def __call__(self, action_onehot: jnp.ndarray, prev_state: jnp.ndarray) -> jnp.ndarray:
        channels = prev_state.shape[-1]
        prev_state = nn.relu(prev_state)

        encoded_action = nn.Dense(features=channels, use_bias=False)(action_onehot)
        x_and_h = prev_state + encoded_action
        out = nn.Sequential([nn.Dense(features=channels, use_bias=False) for _ in range(self.num_blocks)])(x_and_h)
        return out

#! copied this from old codebase utils.py cuz transition needs this
# might consider adding this to the library but not sure which file
def scale_gradient(g: jax.Array, scale: float) -> jax.Array:
    """Scale the gradient.

    Args:
        g (_type_): Parameters that contain gradients.
        scale (float): Scale.

    Returns:
        Array: Parameters with scaled gradients.
    """
    return g * scale + jax.lax.stop_gradient(g) * (1.0 - scale)

#! modified the following from old codebase transition_fn
# I actually don't understand why the original code returned tuple (out, out)
# so to keep it simple I just returned 1 output
class TransitionModule(nn.Module):
    num_actions: int
    num_blocks: int
    scale_grad: float

    def setup(self):
        self.simple_transition = SimpleTransition(num_blocks=self.num_blocks)
    
    def __call__(self, action: int, state: Any) -> Any:
        action_onehot = jax.nn.one_hot(action, 
                                       num_classes=self.num_actions)
        assert action_onehot.ndim in (1, 2), "action_onehot should be [A] or [B, A]"

        # Function to apply the transition and scale the gradient
        def apply_transition(a, s):
            out = self.simple_transition(a, s)
            return scale_gradient(out, self.scale_grad)

        # Check if we are dealing with batched input
        if action_onehot.ndim == 2:
            # Use jax.vmap for vectorized/batched operation
            return jax.vmap(apply_transition)(action_onehot, state)
        else:
            # Apply transition directly for unbatched/single input
            return apply_transition(action_onehot, state)
        
        
#! old codebase did from acme.jax.networks import duelling
# not sure if this still works
# created this in case, assuming that it follows the typical dueling 
# network architecture where it splits the network towards the end 
# into two streams (one for state value and one for action advantages), 
# and then recombines them to produce Q-values
class DuelingMLP(nn.Module):
    num_actions: int
    hidden_sizes: list

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for size in self.hidden_sizes:
            x = nn.Dense(features=size)(x)
            x = nn.relu(x)

        # Separate streams for state value and action advantages
        value = nn.Dense(features=1)(x)
        advantages = nn.Dense(features=self.num_actions)(x)

        # Combine streams: Q = V(s) + A(s, a) - mean(A(s, a))
        q_values = value + advantages - jax.numpy.mean(advantages, axis=1, keepdims=True)
        return q_values
    
#! modified from old codebase prediction_fn
class PredictionModule(nn.Module):
    num_actions: int
    q_dim: int  # old codebase got this from config, not sure where to get this

    @nn.compact
    def __call__(self, state: jax.Array) -> Any:
        # Q-values from Dueling MLP
        # in old codebase it was 
        # q_values = duelling.DuellingMLP(
        # num_actions, hidden_sizes=[config.q_dim])(state)
        q_values = DuelingMLP(num_actions=self.num_actions, hidden_sizes=[self.q_dim])(state)

        # Rewards from a simple MLP, old codebase used
        # rewards = hk.nets.MLP([config.q_dim, 1])(state)
        rewards = nn.Dense(features=self.q_dim)(state)
        rewards = nn.relu(rewards)  # Assuming a ReLU activation before the final layer
        rewards = nn.Dense(features=1)(rewards)

        rewards = jax.numpy.squeeze(rewards, axis=-1)

        # I copied the Prediction structure from contrastiveDyna and put it on the top
        return Predictions(
            state=state, 
            rewards=rewards, 
            q_values=q_values)

#！
class AgentDyna(nn.Module):
    action_dim: int
    hidden_dim: int
    init_scale: float
    num_actions: int  # Number of possible actions
    num_blocks: int  # Number of blocks in SimpleTransition
    scale_grad: float  # Gradient scaling factor
    q_dim: int # old codebase got this from config, not sure where to get this

    def setup(self):

        #? not sure if we need this
        self.rnn = vbb.ScannedRNN(
            hidden_dim=self.hidden_dim,
            cell=self.cell)
        
        #? not sure if we should change this. using a simple MLP for now
        self.observation_encoder = nn.Dense(
            features=self.hidden_dim, 
            kernel_init=nn.initializers.orthogonal(self.init_scale), 
            bias_init=nn.initializers.zeros)

        self.state_fn = DummyRNN() # used this cuz the old codebase used this too

        self.transition_fn = TransitionModule(
            num_actions=self.num_actions,
            num_blocks=self.num_blocks,
            scale_grad=self.scale_grad
        ) # i guess we can use SimpleTransition too?

        self.prediction_fn = PredictionModule(
            #? my understanding is that num_actions is the total number of discrete actions 
            # available for the agent to choose from; action_dim can be used interchangeably?
            num_actions=self.num_actions, #? not sure if this should be num_actions or action_dim
            q_dim=self.q_dim
        )

    #? not sure if we still need this but seems like make_agent will need this for network_params
    def initialize_carry(self, example_shape: Tuple[int]):
        return self.rnn.initialize_carry(example_shape)

    def initialize(self, x: TimeStep):
        """Only used for initialization."""
        # [B, D]
        rnn_state = self.initialize_carry(x.observation.shape)
        return self.__call__(rnn_state, x)
    

    #! network forward pass
    #? I am not sure what should be the inputs here
    def __call__(self, inputs, action, prev_state):
       encoded_obs = self.observation_encoder(inputs)
       state_output, new_state = self.state_fn(encoded_obs, prev_state)
       transition_output = self.transition_fn(action, state_output)
       predictions = self.prediction_fn(transition_output)
       return predictions, new_state

    
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

# PARAMS:
    # config: Configuration dictionary.
    # env: Environment object.
    # env_params: Parameters of the environment.
    # example_timestep: An example of a timestep from the environment.
    # rng: Random number generator key.
# OUTPUT:
    # initializes and returns an agent model along with its param and initial state
def make_agent(
        config: dict,
        env: environment.Environment,
        env_params: environment.EnvParams,
        example_timestep: TimeStep,
        rng: jax.random.KeyArray,
        ) -> Tuple[Agent, Params, AgentState]:

    agent = AgentDyna(
        action_dim=env.action_space(env_params).n,
        hidden_dim=config['AGENT_HIDDEN_DIM'],
        init_scale=config['AGENT_INIT_SCALE'],
        num_actions=config['AGENT_NUM_ACTIONS'],
        num_blocks=config['AGENT_NUM_BLOCKS'],
        scale_grad=config['AGENT_SCALE_GRAD'],
        q_dim=config['AGENT_Q_DIM'],
    )

    rng, _rng = jax.random.split(rng)
    network_params = agent.init(
        _rng, example_timestep, method=agent.initialize)

    init_agent_state = agent.apply(
        network_params,
        example_timestep.observation.shape,
        method=agent.initialize_carry)

    return agent, network_params, init_agent_state

# PARAMS:
    # config (Configuration dictionary with learning rate and other optimizer settings)
# OUTPUT:
    # an optimizer configured based on the provided settings
def make_optimizer(config: dict) -> optax.GradientTransformation:
  def linear_schedule(count):
      frac = 1.0 - (count / config["NUM_UPDATES"])
      return config["LR"] * frac

  lr = linear_schedule if config.get("LR_LINEAR_DECAY", False) else config["LR"]
  return optax.chain(
      optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
      optax.adam(learning_rate=lr, eps=config['EPS_ADAM'])
  )

# PARAMS:
    # config: Configuration dictionary.
    # agent: The agent model.
# OUTPUT:
    # a loss function class
def make_loss_fn_class(config) -> vbb.RecurrentLossFn:
  return functools.partial(
     R2D2LossFn,
     discount=config['GAMMA'])

# OUTPUT:
    # provides actor functions for both training ('actor_step') and evaluation ('eval_step'),
    # handling action selection and prediction steps
def make_actor(config: dict, agent: Agent) -> vbb.Actor:
  explorer = EpsilonGreedy(
      start_e=config["EPSILON_START"],
      end_e=config["EPSILON_FINISH"],
      duration=config["EPSILON_ANNEAL_TIME"]
  )

  def actor_step(
        train_state: vbb.TrainState,
        agent_state: jax.Array,
        timestep: TimeStep,
        rng: jax.random.KeyArray):
    preds, agent_state = agent.apply(
        train_state.params, agent_state, timestep)

    action = explorer.choose_actions(
        preds.q_vals, train_state.timesteps, rng)

    return preds, action, agent_state

  def eval_step(
        train_state: vbb.TrainState,
        agent_state: jax.Array,
        timestep: TimeStep,
        rng: jax.random.KeyArray):
    del rng
    preds, agent_state = agent.apply(
        train_state.params, agent_state, timestep)

    action = preds.q_vals.argmax(-1)

    return preds, action, agent_state

  return vbb.Actor(actor_step=actor_step, eval_step=eval_step)

# all i have to do is define these objects
make_train_preloaded = functools.partial(
   vbb.make_train,
   make_agent=make_agent, # roughly correspondent to the network_factory, which is qlearning.make_minigrid_networks
                          # for contrastive_dyna, it will be contrastive_dyna.make_minigrid_networks
   make_optimizer=make_optimizer, # can use the same one
   make_loss_fn_class=make_loss_fn_class, # should return ContrastiveDynaLossFn
   make_actor=make_actor # same
)