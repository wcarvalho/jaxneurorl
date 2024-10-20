# JaxNeuroRL

This is a library for running RL experiments with some bells and whistles targetting psychologists and neuroscientists.

# Install

## pip install
pip install git+https://github.com/wcarvalho/jaxneurorl.git

## more manauly
conda create -n jaxneurorl python=3.10 pip wheel -y
conda activate jaxneurorl
pip install -r requirements.tx

**if want to run jupyter lab and plot things**
pip install -U jupyterlab matplotlib
<!--1. [FAS Install and Setup](install-fas.md)
2. [Local Install and Setup](install.md)-->

# Single-agent algorithms (all recurrent)

1. [Q-learning](https://openreview.net/forum?id=r1lyTjAqYX) ([code](agents/qlearning.py)). This is a Q-learning based agent.
2. [Universal Successor Feature Approximators](https://arxiv.org/abs/1812.07626) ([code](agents/qlearning.py)). This is a Successor Feature based agent that takes task/policy encodings as inputs.
  1. [Successor Representation](successor_representation.ipynb) - colab explaining the successor representation following [this paper](https://arxiv.org/abs/2402.06590)
2. [AlphaZero](https://arxiv.org/abs/1712.01815) ([code](agents/alphazero.py)). This is a model-based agent that does planning with a ground-truth world model via monte-carlo tree search.







# General todos
1. Replace default LSTM with S5 (maybe [this implementation](https://github.com/facebookresearch/minimax/blob/2ae9e04d37f97d7c14308f5a26237dcfca63470f/src/minimax/models/s5.py#L575)?]
