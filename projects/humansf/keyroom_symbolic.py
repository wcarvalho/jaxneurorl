from typing import Optional
import jax
import jax.numpy as jnp
from projects.humansf.keyroom import KeyRoom, KeyRoomEnvParams, sample_coordinates

from xminigrid.core.constants import Colors, Tiles, DIRECTIONS, TILES_REGISTRY
from gymnax.environments import spaces
from xminigrid.environment import Environment, EnvParams, EnvParamsT
from xminigrid.types import AgentState, EnvCarry, State, TimeStep, EnvCarryT, IntOrArray, StepType, GridState
from xminigrid.core.grid import check_pickable

def get_action_names(maze_config: dict):
    keys = [" ".join(k[::-1]) for k in maze_config['keys']]
    pairs = []
    for p in maze_config['pairs']:
        pairs.extend([" ".join(p[0][::-1]), " ".join(p[1][::-1])])
    return keys + pairs



class SymbolicKeyRoomEnvParams(KeyRoomEnvParams):
    action_objects: jax.Array = None

class KeyRoomSymbolic(KeyRoom):

    def __init__(self, test_end_on_key: bool = False, name='keyroom'):
        super().__init__()
        self.name = name
        self.test_end_on_key = test_end_on_key

    def time_limit(self, params: EnvParams) -> int:
        return 10

    def default_params(self, *args, **kwargs):
        params = super().default_params(*args, **kwargs)
        all_keys = params.maze_config['keys']
        all_pairs = params.maze_config['pairs'].reshape(-1, 2)
        action_objects = jnp.concatenate((all_keys, all_pairs))
        return SymbolicKeyRoomEnvParams(
            **params.__dict__,
            action_objects=action_objects,
        )

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(len(params.action_objects))

    def take_action(self, key, timestep, action, params: SymbolicKeyRoomEnvParams):
        action_object = params.action_objects[action]

        in_grid = (timestep.state.room_grid == action_object[None, None]).all(-1).any()

        grid, agent = jax.lax.cond(
            in_grid,
            lambda: object_action(key, action, action_object, timestep, params),
            lambda: (timestep.state.grid, timestep.state.agent)
        )

        return grid, agent, None

def object_action(rng, action, action_object, timestep, params):
    roomW, roomH = params.width // 3, params.height // 3
    all_room_coords = [
        (1, 2),  # right
        (2, 1),  # bottom
        (1, 0),  # left
        (0, 1),  # top
    ]
    all_room_coords = jnp.array(all_room_coords)
    num_keys = len(params.maze_config['keys'])

    def goto_keys_room():
      """Go to the room that this key maps to."""
      prior_agent = timestep.state.agent
      room_coords = jax.lax.dynamic_index_in_dim(all_room_coords, action, keepdims=False)

      xL = room_coords[1] * roomW
      yT = room_coords[0] * roomH
      x = xL + 3  # center of room
      y = yT + 5  # bottom of room

      agent = AgentState(
          position=jnp.array((y, x), dtype=prior_agent.position.dtype),
          direction=jnp.asarray(0, dtype=prior_agent.direction.dtype),
          pocket=jnp.asarray((Tiles.EMPTY, Colors.EMPTY),
                             dtype=prior_agent.pocket.dtype),
      )
      return timestep.state.grid, agent

    is_key_for_door = action < num_keys
    return jax.lax.cond(
        is_key_for_door,
        lambda: goto_keys_room(),
        lambda: pick_up(timestep.state.grid, timestep.state.agent, action_object)
    )

def pick_up(grid: GridState, agent: AgentState, obj: jnp.array):
    where_present = (grid == obj[None, None]).all(-1).astype(jnp.int32)
    y, x = jnp.nonzero(where_present, size=1)
    next_position = (y[0], x[0])
    #is_pickable = check_pickable(grid, (y, x))
    is_empty_pocket = jnp.equal(agent.pocket[0], Tiles.EMPTY)

    # pick up only if pocket is empty and entity is pickable
    new_grid, new_agent = jax.lax.cond(
        is_empty_pocket,
        lambda: (
            grid.at[next_position[0], next_position[1]].set(
                TILES_REGISTRY[Tiles.FLOOR, Colors.BLACK]),
            agent.replace(
                pocket=TILES_REGISTRY[
                    grid[next_position[0], next_position[1], 0],
                    grid[next_position[0], next_position[1], 1],
                ]
            ),
        ),
        lambda: (grid, agent),
    )
    return new_grid, new_agent
