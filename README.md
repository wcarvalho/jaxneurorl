# Install

# Single-agent algorithms

1. [Q-learning](singleagent/q_learning.py)

**General structure of algorithms**. Each agent is defined by following functions:
```
import functools
from singleagent import value_based_basics as vbb
from gymnax.environments import environment
import jax

def make_agent(
        config: dict,
        env: environment.Environment,
        env_params: environment.EnvParams,
        rng: jax.random.KeyArray,
        ) -> Tuple[Agent, Params, AgentState]:
  """This create a neural network that defines the agent.
      It also, initializes their parameters and initial
      agent state (e.g. LSTM state)"
  ...

def make_optimizer(config: dict) -> optax.GradientTransformation:
  """This create the optimizer (e.g. ADAM) that will optimize 
      the neural network."
  ...

def make_loss_fn_class(config) -> vbb.RecurrentLossFn:
  """This create the loss function that will be used for
      learning (e.g. Q-learning)."
  ...

def make_actor(config: dict, agent: Agent) -> :
  """This creates an Actor with two methods actor_step and eval_step.
      actor_step is used to generate actions during training.
      eval_step is used to generate actions during evaluation.
      For example, with epsilon-greedy Q-learning actor_step 
      maybe sample using the current epsilon-value whereas
      eval_step maybe also select actions according to the
      highest Q-value.""
  ...


custom_make_train = functools.partial(
   vbb.make_train,
   make_agent=make_agent,
   make_optimizer=make_optimizer,
   make_loss_fn_class=make_loss_fn_class,
   make_actor=make_actor)

```



<!-- 2. [Successor Features](td_agents/usfa.py)
3. [MuZero](td_agents/muzero.py) -->


<!-- # Single-agent algorithms -->