from typing import Any, Dict, List, Optional, Sequence, Union


import argparse
import jax
import jax.numpy as jnp
import math
import rlax



class Discretizer:
  def __init__(self,
               max_value: Union[float, int],
               num_bins: Optional[int] = None,
               step_size: Optional[int] = None,
               min_value: Optional[int] = None,
               clip_probs: bool = False,
               tx_pair: rlax.TxPair = rlax.IDENTITY_PAIR):
    self._max_value = max_value
    self._min_value = min_value if min_value is not None else -max_value
    self._clip_probs = clip_probs
    if step_size is None:
      assert num_bins is not None
    else:
      num_bins = math.ceil((self._max_value-self._min_value)/step_size)+1

    self._num_bins = int(num_bins)
    self._tx_pair = tx_pair

  @property
  def num_bins(self):
    return self._num_bins

  def logits_to_scalar(self, logits):
     return self.probs_to_scalar(jax.nn.softmax(logits))

  def probs_to_scalar(self, probs):
    scalar = rlax.transform_from_2hot(
      probs=probs,
      min_value=self._min_value,
      max_value=self._max_value,
      num_bins=self._num_bins)
    unscaled_scalar = self._tx_pair.apply_inv(scalar)
    return unscaled_scalar

  def scalar_to_probs(self, scalar):
      scaled_scalar = self._tx_pair.apply(scalar)
      probs = rlax.transform_to_2hot(
      scalar=scaled_scalar,
      min_value=self._min_value,
      max_value=self._max_value,
      num_bins=self._num_bins)
      probs = jnp.clip(probs, 0, 1)  # for numerical stability
      return probs


def make_parser():
    parser = argparse.ArgumentParser(description="")

    #parser.add_argument("--program", type=str, help="")
    #parser.add_argument("--config_path", type=str, help="")

    # Sweep settings
    parser.add_argument('--parallel', type=str, default='none',
                        help="none: run 1 experiment. sbatch: run many experiments with SBATCH. ray: run many experiments with ray. Use sbatch with SLURM or ray otherwise.")

    parser.add_argument("--search_method", type=str, default='bayes', help="")
    parser.add_argument('--debug_sweep', type=bool, default=False,
                        help='whether to use wandb.')
    
    # Run settings
    parser.add_argument('--debug', type=bool, default=False,
                        help='whether to use wandb.')
    parser.add_argument("--search", type=str, help="")
    parser.add_argument('--wandb', type=bool, default=True,
                        help='whether to use wandb.')

    # Resources
    parser.add_argument('--num_gpus', type=int,
                        default=1, help='number of GPUs.')
    parser.add_argument('--num_cpus', type=int,
                        default=16, help='number of CPUs.')
    parser.add_argument('--memory', type=int,
                        default=120000, help='memory in MBs.')
    parser.add_argument('--time', type=str,
                        default='0-01:00:00', help='1 hour.')
    parser.add_argument('--max_concurrent', type=int,
                        default=16, help='number of concurrent jobs.')

    parser.add_argument('--account', type=str, default='',
                        help='account on slurm servers to use.')
    parser.add_argument('--partition', type=str, default='kempner',
                        help='partition on slurm servers to use.')

    # DO NOT USE BELOW. Set by parallel.py
    parser.add_argument('--settings_config', type=str, default='',
                        help='config file produced by parallel')
    parser.add_argument('--subprocess', action='store_true',
                        help='label for whether this run is a subprocess.')

    return parser

