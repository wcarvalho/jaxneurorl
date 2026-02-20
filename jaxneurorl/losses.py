from typing import Optional
import chex
import jax
import rlax


def sarsa_lambda(
  q_tm1: jax.Array,
  a_tm1: jax.Array,
  r_t: jax.Array,
  discount_t: jax.Array,
  q_t: jax.Array,
  a_t: jax.Array,
  lambda_: jax.Array,
  stop_target_gradients: bool = True,
) -> jax.Array:
  """Calculates the SARSA(lambda) temporal difference error.

  See "Reinforcement Learning: An Introduction" by Sutton and Barto.
  (http://incompleteideas.net/book/ebook/node77.html).

  Args:
    q_tm1: sequence of Q-values at time t-1.
    a_tm1: sequence of action indices at time t-1.
    r_t: sequence of rewards at time t.
    discount_t: sequence of discounts at time t.
    q_t: sequence of Q-values at time t.
    a_t: sequence of action indices at time t.
    lambda_: mixing parameter lambda, either a scalar or a sequence.
    stop_target_gradients: bool indicating whether or not to apply stop gradient
      to targets.

  Returns:
    SARSA(lambda) temporal difference error.
  """
  chex.assert_rank(
    [q_tm1, a_tm1, r_t, discount_t, q_t, a_t, lambda_], [2, 1, 1, 1, 2, 1, {0, 1}]
  )
  chex.assert_type(
    [q_tm1, a_tm1, r_t, discount_t, q_t, a_t, lambda_],
    [float, int, float, float, float, int, float],
  )

  qa_tm1 = rlax.batched_index(q_tm1, a_tm1)
  qa_t = rlax.batched_index(q_t, a_t)
  target_tm1 = rlax.lambda_returns(r_t, discount_t, qa_t, lambda_)
  target_tm1 = jax.lax.select(
    stop_target_gradients, jax.lax.stop_gradient(target_tm1), target_tm1
  )

  return qa_tm1, target_tm1


def q_learning_lambda_target(
  q_t: jax.Array,
  r_t: jax.Array,
  discount_t: jax.Array,
  is_last_t: jax.Array,
  a_t: jax.Array,
  lambda_: jax.Array,
  stop_target_gradients: bool = True,
) -> jax.Array:
  """MINOR change to rlax.lambda_returns to incorporate is_last_t.

  lambda_ = jax.numpy.ones_like(discount_t) * lambda_ * (1 - is_last_t).
                                            ONLY CHANGE:^
  """
  v_t = rlax.batched_index(q_t, a_t)
  lambda_ = jax.numpy.ones_like(discount_t) * lambda_ * (1 - is_last_t)
  target_tm1 = rlax.lambda_returns(
    r_t, discount_t, v_t, lambda_, stop_target_gradients=stop_target_gradients
  )
  return target_tm1


def q_learning_lambda_td(
  q_tm1: jax.Array,
  a_tm1: jax.Array,
  target_q_t: jax.Array,
  a_t: jax.Array,
  r_t: jax.Array,
  discount_t: jax.Array,
  is_last_t: jax.Array,
  lambda_: jax.Array,
  stop_target_gradients: bool = True,
  tx_pair: rlax.TxPair = rlax.IDENTITY_PAIR,
):
  """Essentially the same as rlax.q_lambda except we use selector actions on q-values, not average. This makes it like Q-learning.

  Other difference is is_last_t is here.
  """

  # Apply signed hyperbolic transform to Q-values
  q_tm1_transformed = tx_pair.apply(q_tm1)
  target_q_t_transformed = tx_pair.apply(target_q_t)

  v_tm1 = rlax.batched_index(q_tm1_transformed, a_tm1)
  target_mt1 = q_learning_lambda_target(
    r_t=r_t,
    q_t=target_q_t_transformed,
    a_t=a_t,
    discount_t=discount_t,
    is_last_t=is_last_t,
    lambda_=lambda_,
    stop_target_gradients=stop_target_gradients,
  )

  v_tm1, target_mt1 = tx_pair.apply_inv(v_tm1), tx_pair.apply_inv(target_mt1)

  return v_tm1, target_mt1


def n_step_target(
  v_t: jax.Array,
  r_t: jax.Array,
  discount_t: jax.Array,
  is_last_t: jax.Array,
  lambda_: jax.Array,
  n: int = 5,
  stop_target_gradients: bool = True,
):
  lambda_ = jax.numpy.ones_like(discount_t) * lambda_ * (1 - is_last_t)
  return rlax.n_step_bootstrapped_returns(
    r_t=r_t,
    discount_t=discount_t,
    v_t=v_t,
    lambda_t=lambda_,
    n=n,
    stop_target_gradients=stop_target_gradients,
  )


def q_learning_n_step_td(
  q_tm1: jax.Array,
  a_tm1: jax.Array,
  target_q_t: jax.Array,
  a_t: jax.Array,
  r_t: jax.Array,
  discount_t: jax.Array,
  is_last_t: jax.Array,
  lambda_: jax.Array,
  n: int = 5,
  stop_target_gradients: bool = True,
):
  """Essentially the same as rlax.n_step_q_learning.
  Only difference is is_last_t is here.
  """
  q_a_t = rlax.batched_index(target_q_t, a_t)
  target_mt1 = n_step_target(
    r_t=r_t,
    v_t=q_a_t,
    discount_t=discount_t,
    is_last_t=is_last_t,
    lambda_=lambda_,
    n=n,
    stop_target_gradients=stop_target_gradients,
  )
  q_a_tm1 = rlax.batched_index(q_tm1, a_tm1)
  return q_a_tm1, target_mt1


def cql_loss(
  q_vals: jax.Array,  # [T, A]
  actions: jax.Array,  # [T]
  temperature: float = 1.0,
) -> jax.Array:
  """Conservative Q-Learning regularizer for discrete actions.
  Returns per-timestep CQL penalty: temp * logsumexp(Q/temp) - Q(s, a_data).
  """
  q_data = rlax.batched_index(q_vals, actions)
  logsumexp_q = temperature * jax.scipy.special.logsumexp(q_vals / temperature, axis=-1)
  return logsumexp_q - q_data
