# Install

# Single-agent algorithms

1. [Q-learning](singleagent/q_learning.py)

**General structure of algorithms**. Each agent is defined by following functions:
```
def make_agent(
        config: dict,
        env: environment.Environment,
        env_params: environment.EnvParams,
        rng: jax.random.KeyArray,
        ) -> Tuple[Agent, Params, AgentState]:
  ...

def make_optimizer(config: dict) -> optax.GradientTransformation:
  ...

def make_loss_fn_class(config):
  ...

def make_actor(config: dict, agent: Agent):
  ...

import functools
from singleagent import value_based_basics as vbb

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