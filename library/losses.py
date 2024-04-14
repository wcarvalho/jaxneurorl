from typing import Optional
import chex
import jax
import rlax


#def lambda_returns(
#    r_t: jax.Array,
#    discount_t: jax.Array,
#    v_t: jax.Array,
#    is_last_t: Optional[jax.Array]=None,
#    lambda_: chex.Numeric = 1.,
#    stop_target_gradients: bool = False,
#) -> jax.Array:
#  if is_last_t is None:
#    is_last_t = jax.numpy.zeros_like(discount_t)
#  chex.assert_rank([r_t, discount_t, v_t, is_last_t,
#                   lambda_], [1, 1, 1, 1, {0, 1}])
#  chex.assert_type([r_t, discount_t, v_t, is_last_t, lambda_], float)
#  chex.assert_equal_shape([r_t, discount_t, v_t, is_last_t])

#  # If scalar make into vector.
#  lambda_ = jax.numpy.ones_like(discount_t) * lambda_ * (1 - is_last_t)

#  # Work backwards to compute `G_{T-1}`, ..., `G_0`.
#  def _body(acc, xs):
#    returns, discounts, values, lambda_ = xs
#    acc = returns + discounts * ((1-lambda_) * values + lambda_ * acc)
#    return acc, acc

#  _, returns = jax.lax.scan(
#      _body, v_t[-1], (r_t, discount_t, v_t, lambda_), reverse=True)

#  return jax.lax.select(stop_target_gradients,
#                        jax.lax.stop_gradient(returns),
#                        returns)

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
  target_tm1 = rlax.lambda_returns(r_t, discount_t, v_t, lambda_)

  target_tm1 = jax.lax.select(stop_target_gradients,
                              jax.lax.stop_gradient(target_tm1), target_tm1)
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
    stop_target_gradients: bool = True):
    """Essentially the same as rlax.q_lambda except we use selector actions on q-values, not average.
      This makes it like Q-learning.
    """

    v_tm1 = rlax.batched_index(q_tm1, a_tm1)
    target_mt1 = q_learning_lambda_target(
        r_t=r_t, q_t=target_q_t, a_t=a_t,
        discount_t=discount_t,
        is_last_t=is_last_t,
        lambda_=lambda_,
        stop_target_gradients=stop_target_gradients)
    return v_tm1, target_mt1
