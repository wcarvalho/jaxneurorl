
# Single-agent algorithms (all recurrent)

1. [Q-learning](https://openreview.net/forum?id=r1lyTjAqYX) ([code](singleagent/qlearning.py)). This is a Q-learning based agent.
2. [AlphaZero](https://arxiv.org/abs/1712.01815) ([code](singleagent/alphazero.py)). This is a model-based agent that does planning with a ground-truth world model via monte-carlo tree search.

**General structure of algorithms**. Each agent is defined by functions for creating:
- the agent's neural network
- the neural network's optimizer
- the agent's loss function/objective
- the agent's actor class which selects action in response to observations

These functions are given as input to a `make_train` function which creates a `train` function that runs the experiment.

Below is a schematic of the general learning algorithm used in this [codebase](singleagent/value_based_basics.py).
<img src="images/overview.png" alt="FARM" style="zoom:40%;" />



# Install


1. [FAS Install and Setup](install-fas.md)
2. [Local Install and Setup](install.md)


# General todos
1. Replace default LSTM with S5 (maybe [this implementation](https://github.com/facebookresearch/minimax/blob/2ae9e04d37f97d7c14308f5a26237dcfca63470f/src/minimax/models/s5.py#L575)?]
