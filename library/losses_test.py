# This file should be named test_losses.py (or similar)

from typing import Optional
import chex
import jax.numpy as jnp
from library import losses  # Assuming losses.py is in a folder named 'library'
import rlax


class TestLosses:  # Define a class to hold the test function
  def test_q_learning_lambda_td(self):
    """Tests q_learning_lambda_td function for different scenarios."""
    pairs = [
        ('terminal', True, 0),
        ('truncation', True, 1),
        #('step', False, 1),
    ]
    gamma = .99
    lambda__ = .9
    for loss_fn in (losses.q_learning_lambda_td, losses.q_learning_n_step_td):
      for t, terminal, discount in pairs:

        a = jnp.array([0, 0, 0], dtype=jnp.int32)
        q = jnp.array([[0, 0, 0],
                      [0, 0, 0]], dtype=jnp.float32).T  # make time first
        r = jnp.array([0, 0, 1], dtype=jnp.float32)
        discount = jnp.array([1, 1, discount], dtype=jnp.float32)
        is_last = jnp.array([0, 0, terminal], dtype=jnp.float32)
        lambda_ = jnp.array([1, 1, 1], dtype=jnp.float32)

        q = jnp.concatenate((q, q))
        a = jnp.concatenate((a, a))
        r = jnp.concatenate((r, r))
        discount = jnp.concatenate((discount, discount))
        is_last = jnp.concatenate((is_last, is_last))
        lambda_ = jnp.concatenate((lambda_, lambda_))

        v_tm1, target_mt1 = loss_fn(
            q_tm1=q[:-1],
            a_tm1=a[:-1],
            target_q_t=q[1:],
            a_t=a[1:],
            r_t=r[1:],
            discount_t=discount[1:] * gamma,
            is_last_t=is_last[1:],
            lambda_=lambda_[1:] * lambda__,
        )
        target_mt1 = target_mt1*discount[:-1]

        loss = target_mt1 - v_tm1
        truncated = ((discount+is_last) > 1)

        loss_mask = (1-truncated).astype(loss.dtype)
        loss = loss*(loss_mask[:-1])

        # Tolerance for floating-point comparison
        atol = 1e-6

        print(loss_fn, t, target_mt1)
        if t == 'terminal':
          # Check target values with chex.check_close
          chex.assert_trees_all_close(
            target_mt1[:2],
            target_mt1[-2:],
            atol=atol)
          chex.assert_trees_all_close(
              target_mt1[2], 0, atol=atol)
        elif t == 'truncation':
          chex.assert_trees_all_close(
              target_mt1[:2],
              target_mt1[-2:],
              atol=atol)


# Run tests if called directly (e.g., python test_losses.py)
if __name__ == "__main__":
  # Assuming you have a test runner like pytest, use it here
  # For example, with pytest:
   import pytest
   pytest.main()
