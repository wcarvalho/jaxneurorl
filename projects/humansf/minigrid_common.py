from __future__ import annotations

import functools
import jax
import jax.numpy as jnp
from typing_extensions import TypeAlias
from flax import struct
import jax.numpy as jnp

from xminigrid.types import AgentState, GridState, IntOrArray
from xminigrid.core.constants import DIRECTIONS, TILES_REGISTRY, Colors, Tiles, NUM_TILES, NUM_COLORS
from xminigrid.core.grid import check_can_put, check_pickable, check_walkable, equal

from xminigrid.core.constants import Tiles
from xminigrid.wrappers import Wrapper

NUM_DIRECTIONS = 4

def make_binary_vector(obj):
    binary_vector = jnp.zeros(NUM_TILES + NUM_COLORS)

    # Extract the category and color vectors from the obj
    category_idx = obj[0]
    color_idx = obj[1]

    # Set the corresponding indices in the binary vector to 1
    binary_vector = binary_vector.at[category_idx].set(1)
    binary_vector = binary_vector.at[NUM_TILES + color_idx].set(1)

    return binary_vector

make_binary_vector_grid = jax.vmap(jax.vmap(make_binary_vector))

def position_to_two_hot(local_agent_position, grid_shape):
    # Extract the position and grid dimensions
    y, x = local_agent_position
    max_y, max_x = grid_shape

    # Initialize one-hot vectors
    one_hot_x = jnp.zeros(max_x)
    one_hot_y = jnp.zeros(max_y)
    
    # Set the corresponding positions to 1
    one_hot_x = one_hot_x.at[x].set(1)
    one_hot_y = one_hot_y.at[y].set(1)

    return jnp.concatenate((one_hot_x, one_hot_y))

def make_agent_grid(grid_shape, agent_pos, dir):
    height, width = grid_shape


    # Create binary grid with a 1 at the agent's position
    binary_grid = jnp.zeros((height, width, 1))
    binary_grid = binary_grid.at[agent_pos[0], agent_pos[1]].set(1)

    # Create binary vector with a 1 at the agent's direction
    binary_direction = jnp.zeros((height, width, NUM_DIRECTIONS))
    binary_direction = binary_direction.at[agent_pos[0], agent_pos[1], dir].set(
        1)

    # Combine binary grid and direction vector
    agent_grid = jnp.concatenate((binary_grid, binary_direction), axis=-1)

    return agent_grid

def prepare_grid(grid, extra_dims):
    padding_shape = (*grid.shape[:2], extra_dims)
    padding = jnp.zeros(padding_shape, dtype=grid.dtype)
    return jnp.concatenate((grid, padding), axis=-1)


class States(struct.PyTreeNode):
    ON_FLOOR: int = struct.field(pytree_node=False, default=0)
    PICKED_UP: int = struct.field(pytree_node=False, default=1)


def make_obj(
        category: int,
        color: int,
        visible: int = 0,
        state: int = States.ON_FLOOR,
        asarray: bool = False):
    obj = (category, color, visible, state)
    if asarray:
        obj = jnp.asarray(obj, dtype=jnp.uint8)
    return obj


#class AutoResetWrapper:

#    def __init__(self, env: Environment[EnvParamsT, EnvCarryT]):
#        self._env = env

#    def __auto_reset(self, key, params, timestep):
#        key, key_ = jax.random.split(key)
#        return self._env.reset(key_, params)

#    def step(self,
#             key: jax.random.PRNGKey,
#             prior_timestep,
#             action,
#             params):
#        return jax.lax.cond(
#            prior_timestep.last(),
#            lambda: self.__auto_reset(key, params, prior_timestep),
#            lambda: self._env.step(key, prior_timestep, action, params),
#        )
