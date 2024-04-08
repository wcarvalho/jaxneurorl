
# Single-agent algorithms

1. [Q-learning](singleagent/qlearning.py)

**General structure of algorithms**. Each agent is defined by functions for creating:
- the agent's neural network
- the neural network's optimizer
- the agent's loss function/objective
- the agent's actor class which selects action in response to observations

These functions are given as input to a `make_train` function which creates a `train` function that runs the experiment.

Below is a schematic of the general learning algorithm used in this codebase.
<img src="images/overview.png" alt="FARM" style="zoom:40%;" />



# Install


1. [FAS Install and Setup](install-fas.md)
2. [Local Install and Setup](install.md)
