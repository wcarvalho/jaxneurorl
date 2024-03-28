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

make_obj_arr = functools.partial(make_obj, asarray=True)
ActionOutput: TypeAlias = tuple[GridState, AgentState, jax.Array]


def _move(position: jax.Array, direction: jax.Array) -> jax.Array:
    direction = jax.lax.dynamic_index_in_dim(
        DIRECTIONS, direction, keepdims=False)
    new_position = position + direction
    return new_position


def move_forward(grid: GridState, agent: AgentState) -> ActionOutput:
    next_position = jnp.clip(
        _move(agent.position, agent.direction),
        a_min=jnp.array((0, 0)),
        a_max=jnp.array((grid.shape[0] - 1, grid.shape[1] - 1)),  # H, W
    )
    position = jax.lax.select(
        check_walkable(grid, next_position),
        next_position,
        agent.position,
    )
    new_agent = agent.replace(position=position)

    return grid, new_agent, new_agent.position


def turn_clockwise(grid: GridState, agent: AgentState) -> ActionOutput:
    new_direction = (agent.direction + 1) % 4
    new_agent = agent.replace(direction=new_direction)
    return grid, new_agent, agent.position


def turn_counterclockwise(grid: GridState, agent: AgentState) -> ActionOutput:
    new_direction = (agent.direction - 1) % 4
    new_agent = agent.replace(direction=new_direction)
    return grid, new_agent, agent.position


def pick_up(grid: GridState, agent: AgentState) -> ActionOutput:

    next_position = _move(agent.position, agent.direction)

    is_pickable = check_pickable(grid, next_position)
    is_empty_pocket = jnp.equal(agent.pocket[0], Tiles.EMPTY)
    # pick up only if pocket is empty and entity is pickable
    new_grid, new_agent = jax.lax.cond(
        is_pickable & is_empty_pocket,
        lambda: (
            grid.at[next_position[0], next_position[1]].set(
                make_obj(Tiles.FLOOR, Colors.BLACK)),
            agent.replace(
                pocket=make_obj_arr(*grid[next_position[0], next_position[1], :2],
                                    visible=1, state=States.PICKED_UP),
            ),
        ),
        lambda: (grid, agent),
    )
    return new_grid, new_agent, next_position


def put_down(grid: GridState, agent: AgentState) -> ActionOutput:
    next_position = _move(agent.position, agent.direction)

    can_put = check_can_put(grid, next_position)
    not_empty_pocket = jnp.not_equal(agent.pocket[0], Tiles.EMPTY)
    new_grid, new_agent = jax.lax.cond(
        can_put & not_empty_pocket,
        lambda: (
            grid.at[next_position[0], next_position[1]].set(agent.pocket),
            agent.replace(pocket=make_obj_arr(Tiles.EMPTY, Colors.EMPTY)),
        ),
        lambda: (grid, agent),
    )
    return new_grid, new_agent, next_position


# TODO: may be this should be open_door action? toggle is too general and box is not supported yet
def toggle(grid: GridState, agent: AgentState) -> ActionOutput:
    next_position = _move(agent.position, agent.direction)
    next_tile = grid[next_position[0], next_position[1]]

    # check door_locked
    new_grid = jax.lax.select(
        jnp.equal(next_tile[0], Tiles.DOOR_LOCKED) & equal(
            agent.pocket[:2], TILES_REGISTRY[Tiles.KEY, next_tile[1]]),
        grid.at[next_position[0], next_position[1]].set(
            make_obj(Tiles.DOOR_OPEN, next_tile[1])),
        grid,
    )
    # check door_closed
    new_grid = jax.lax.select(
        jnp.equal(next_tile[0], Tiles.DOOR_CLOSED),
        grid.at[next_position[0], next_position[1]].set(
            make_obj(Tiles.DOOR_OPEN, next_tile[1])),
        new_grid,
    )
    # check door_open
    new_grid = jax.lax.select(
        jnp.equal(next_tile[0], Tiles.DOOR_OPEN),
        grid.at[next_position[0], next_position[1]].set(
            make_obj(Tiles.DOOR_CLOSED, next_tile[1])),
        new_grid,
    )
    return new_grid, agent, next_position


def take_action(grid: GridState, agent: AgentState, action: IntOrArray) -> ActionOutput:
    # This will evaluate all actions.
    # Can we fix this and choose only one function? It'll speed everything up dramatically.
    actions = (
        lambda: move_forward(grid, agent),
        lambda: turn_clockwise(grid, agent),
        lambda: turn_counterclockwise(grid, agent),
        lambda: pick_up(grid, agent),
        lambda: put_down(grid, agent),
        lambda: toggle(grid, agent),
    )
    new_grid, new_agent, changed_position = jax.lax.switch(action, actions)

    return new_grid, new_agent, changed_position


# gym and gymnasium style reset (on the same step with termination)
class AutoResetWrapper(Wrapper):
    def __auto_reset(self, key, params, timestep):
        key, key_ = jax.random.split(key)
        return self._env.reset(key_, params)


    # TODO: add last_obs somewhere in the timestep? add extras like in Jumanji?
    def step(self,
             key: jax.random.KeyArray,
             prior_timestep,
             action,
             params):
        return jax.lax.cond(
            prior_timestep.last(),
            lambda: self.__auto_reset(params, prior_timestep),
            lambda: self._env.step(key, prior_timestep, action, params),
        )
