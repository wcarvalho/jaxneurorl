from typing import Optional
import chex
import jax
import rlax

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
  target_tm1 = rlax.lambda_returns(r_t, discount_t, v_t, lambda_,
                                   stop_target_gradients=stop_target_gradients)
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
        stop_target_gradients=stop_target_gradients)

    v_tm1, target_mt1 = tx_pair.apply_inv(v_tm1), tx_pair.apply_inv(target_mt1)

    return v_tm1, target_mt1

def n_step_target(
        v_t: jax.Array,
        r_t: jax.Array,
        discount_t: jax.Array,
        is_last_t: jax.Array,
        lambda_: jax.Array,
        n: int = 5,
        stop_target_gradients: bool = True
):
   lambda_ = jax.numpy.ones_like(discount_t) * lambda_ * (1 - is_last_t)
   return rlax.n_step_bootstrapped_returns(
      r_t=r_t,
      discount_t=discount_t,
      v_t=v_t,
      lambda_t=lambda_,
      n=n,
      stop_target_gradients=stop_target_gradients
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
        stop_target_gradients: bool = True):
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
        stop_target_gradients=stop_target_gradients)
    q_a_tm1 = rlax.batched_index(q_tm1, a_tm1)
    return q_a_tm1, target_mt1
