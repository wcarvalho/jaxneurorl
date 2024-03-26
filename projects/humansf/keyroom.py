"""

Grid features:
- Category
- Color
- Visible
- State
"""
from __future__ import annotations

from functools import partial
from typing import Optional, Generic, Callable
import jax
import jax.numpy as jnp
from flax import struct
import numpy as np
import copy
from gymnax.environments import spaces

from xminigrid.core.constants import Colors, Tiles, NUM_ACTIONS
from xminigrid.core.goals import EmptyGoal
from xminigrid.core.actions import take_action
from xminigrid.core.grid import cartesian_product_1d, nine_rooms, sample_coordinates, sample_direction, free_tiles_mask
from xminigrid.rendering.rgb_render import render
from xminigrid.rendering.text_render import render as text_render
from xminigrid.core.rules import EmptyRule
from xminigrid.environment import Environment, EnvParams, EnvParamsT
from xminigrid.types import AgentState, EnvCarry, State, TimeStep, EnvCarryT, IntOrArray, StepType, GridState

from projects.humansf import minigrid_common
from projects.humansf.minigrid_common import States
from projects.humansf.minigrid_common import make_obj as make_task_obj
# from projects.humansf.minigrid_common import take_action as take_action_new

KEY_IDX = 0
DOOR_IDX = 1
TRAIN_OBJECT_IDX = 2
TEST_OBJECT_IDX = 3

TYPE_IDX = 0
COLOR_IDX = 1

def make_obj(
        category: int,
        color: int,
        visible: int = 0,
        state: int = States.ON_FLOOR,
        asarray: bool = False):
    del visible
    del state
    obj = (category, color)
    if asarray:
        obj = jnp.asarray(obj, dtype=jnp.uint8)
    return obj

make_obj_arr = partial(make_obj, asarray=True)

def accomplished(grid, task):
  # [D] [H, W, D]
  # was this task accomplioshed
  accomplished_somewhere = (task[None, None]==grid).all(axis=-1)
  accomplished_anywhere = accomplished_somewhere.any()
  return accomplished_anywhere

accomplished_H = jax.vmap(accomplished, in_axes=(None, 0))
accomplished_HW = jax.vmap(accomplished_H, in_axes=(None, 0))

def shorten_maze_config(maze_config: dict, n: int):
  maze_config = copy.deepcopy(maze_config)
  maze_config['keys'] = maze_config['keys'][:n]
  maze_config['pairs'] = maze_config['pairs'][:n]
  return maze_config

class TaskState(struct.PyTreeNode):
   feature_counts: jax.Array
   features: jax.Array

class TaskRunner(struct.PyTreeNode):
  """_summary_

  members:
      task_objects (jax.Array): [num_tasks, num_task_objects, task_object_dim]

  Returns:
      _type_: _description_
  """
  task_objects: jax.Array
  first_instance: bool = True
  convert_type: Callable[[jax.Array], jax.Array] = lambda x: x.astype(jnp.int32)

  def get_features(self, visible_grid: GridState, agent: AgentState):
    """Get features
    `task_objects` is  [num_tasks, num_task_objects, task_object_dim]
    `features` returned will also be [num_tasks, num_task_objects]

    we're doing to look for those task objects in every grid position.
      if the task object is found in ANY grid position or in agent.pocket,
      features[task, task_object] = 1, else 0.

    Args:
        visible_grid (GridState): [H, W, 2]
        agent (AgentState): _description_
    """
    def add_visibility(grid):
        padding_shape = (*grid.shape[:2], 1)
        padding = jnp.ones(padding_shape, dtype=grid.dtype)
        return jnp.concatenate((grid, padding), axis=-1)

    # add dimension with all 1s, since each position here is visible to agent
    visible_grid = add_visibility(visible_grid)  

    # acc_in_grid is [num_tasks, num_task_objects]
    acc_in_grid = accomplished_HW(visible_grid, self.task_objects[:, :, 3])

    # look at all task objects and see if they match pocket
    # acc_in_pocket is [num_tasks, num_task_objects]
    pocket = make_task_obj(*agent.pocket, visible=1, state=States.PICKED_UP, asarray=True)
    acc_in_pocket = (pocket[None, None]==self.task_objects).all(axis=-1)

    acc = jnp.logical_or(acc_in_grid, acc_in_pocket)

    features = self.convert_type(acc)
    return features

  def reset(self, visible_grid: GridState, agent: AgentState):
    """Get initial features.

    Args:
        visible_grid (GridState): _description_
        agent (AgentState): _description_

    Returns:
        _type_: _description_
    """
    features = self.get_features(visible_grid, agent)
    return TaskState(
       feature_counts=features,
       features=features
    )

  def step(self, prior_state: TaskState, visible_grid: GridState, agent: AgentState):
    features = self.get_features(visible_grid, agent)
    difference = self.convert_type(features - prior_state.features)
    positive_difference = difference*(difference > 0)

    feature_counts = prior_state.feature_counts + positive_difference
    if self.first_instance:
      # in 1 setting, state-feature is only active the 1st time the count goes  0 -> +
      # print('feature_counts == 1', feature_counts == 1)
      is_first = self.convert_type(feature_counts == 1)
      new_features = is_first*positive_difference
    else:
      # in this setting, whenever goes 0 -> +
      new_features = positive_difference

    return TaskState(
        feature_counts=feature_counts,
        features=new_features
    )

class KeyRoomEnvParams(EnvParams):
    # num_objects: int = struct.field(pytree_node=False, default=12)
    random_door_loc: bool = struct.field(pytree_node=False, default=False)
    training: bool = struct.field(pytree_node=False, default=True)

class EnvState(struct.PyTreeNode, Generic[EnvCarryT]):
    key: jax.Array
    step_num: jax.Array

    grid: GridState
    room_grid: GridState
    agent: AgentState
    local_agent_position: jax.Array
    # task: Task
    carry: EnvCarryT
    feature_weights: jax.Array
    goal_room_idx: jax.Array
    task_state: Optional[TaskState] = None

def convert_dict_to_types(maze_dict):
    # Create a dictionary to map color strings to their corresponding values

    color_map = {
        "red": Colors.RED,
        "green": Colors.GREEN,
        "blue": Colors.BLUE,
        "purple": Colors.PURPLE,
        "yellow": Colors.YELLOW,
        "grey": Colors.GREY,
    }

    # Create a dictionary to map object strings to their corresponding tile values
    object_map = {
        "key": Tiles.KEY,
        "box": Tiles.SQUARE,
        "ball": Tiles.BALL,
    }
    # Convert keys
    converted_keys = []
    for key_pair in maze_dict["keys"]:
        obj, color = key_pair
        converted_key = [object_map[obj], color_map[color]]
        converted_keys.append(converted_key)

    # Convert pairs
    converted_pairs = []
    for pair in maze_dict["pairs"]:
        converted_pair = []
        for obj_color in pair:
            obj, color = obj_color
            converted_obj_color = [object_map[obj], color_map[color]]
            converted_pair.append(converted_obj_color)
        converted_pairs.append(converted_pair)

    # Create the converted dictionary
    converted_dict = {
        "keys": jnp.array(converted_keys, dtype=jnp.uint8),
        "pairs": jnp.array(converted_pairs, dtype=jnp.uint8),
    }

    return converted_dict


def get_room_grid(grid: GridState, agent: AgentState):
    agent_pos = agent.position
    grid_width = grid.shape[0]
    grid_dim = grid.shape[-1]

    # Calculate the room size
    # NOTE: ASSUME SQUARE ROOMS
    room_size = grid_width // 3

    # Calculate the room indices
    room_x = agent_pos[0] // room_size
    room_y = agent_pos[1] // room_size

    # Calculate the starting and ending coordinates of the room
    start_x = room_x * room_size
    start_y = room_y * room_size

    delta = grid_width//3 + 1

    return jax.lax.dynamic_slice(
        grid, (start_x, start_y, 0), (delta, delta, grid_dim))

def get_local_agent_position(agent_pos, height, width):
    # Calculate the room size
    room_height = height // 3
    room_width = width // 3
    
    # Calculate the room indices
    room_x = agent_pos[0] // room_width
    room_y = agent_pos[1] // room_height
    
    # Calculate the starting coordinates of the room
    start_x = room_x * room_width
    start_y = room_y * room_height
    
    # Calculate the local position within the room
    local_x = agent_pos[0] - start_x
    local_y = agent_pos[1] - start_y
    
    return jnp.array([local_x, local_y])

class Observation(struct.PyTreeNode):
   image: jax.Array
   task_w: jax.Array
   state_features: jax.Array
   has_occurred: jax.Array
   pocket: jax.Array

def render_room(state: EnvState, render_mode: str = "rgb_array", **kwargs):
  room_grid = np.asarray(state.room_grid)
  local_agent_position = get_local_agent_position(
     state.agent.position,
     *state.grid.shape[:2])
  localized_agent = state.agent.replace(
      position=local_agent_position)
  if render_mode == "rgb_array":
    return render(room_grid, localized_agent, **kwargs)
  elif render_mode == "rich_text":
    return text_render(room_grid, localized_agent, **kwargs)
  else: 
     raise NotImplementedError(render_mode)

def prepare_task_variables(maze_config: struct.PyTreeNode):
  keys = maze_config['keys']
  pairs = maze_config['pairs']
  n_task_rooms = len(keys)
  task_objects = []
  train_w = []
  test_w = []
  for room_idx in range(n_task_rooms):
    goal_key = make_task_obj(*keys[room_idx], visible=1, state=States.PICKED_UP)
    goal_door = make_task_obj(Tiles.DOOR_OPEN, keys[room_idx][COLOR_IDX], visible=1)

    obj1, obj2 = pairs[room_idx]
    train_object = make_task_obj(*obj1, visible=1, state=States.PICKED_UP)
    test_object = make_task_obj(*obj2, visible=1, state=States.PICKED_UP)

    task_objects.append((goal_key, goal_door, train_object, test_object))
    train_w.append((.1, .25, 1., 0))
    test_w.append((0., 0., 0., 1.0))

  task_objects = jnp.array(task_objects)
  train_w = jnp.array(train_w)
  test_w = jnp.array(test_w)

  return task_objects, train_w, test_w


def make_observation(state: EnvState, room_grid: GridState):
  """This converts all inputs into binary vectors. this faciitates processing with a neural network."""
  binary_room_grid = minigrid_common.make_binary_vector_grid(room_grid)
  agent_grid = minigrid_common.make_agent_grid(
      grid_shape=room_grid.shape[:2],  # [H,w]
      agent_pos=state.local_agent_position,  # [y, x]
      dir=state.agent.direction,  # [d]
  )

  return Observation(
      image=jnp.concatenate((binary_room_grid, agent_grid), axis=-1),
      pocket=minigrid_common.make_binary_vector(state.agent.pocket),
      state_features=state.task_state.features.flatten(),
      has_occurred=(state.task_state.feature_counts >= 1).flatten(),
      task_w=state.feature_weights.flatten(),
      )

class KeyRoom(Environment[KeyRoomEnvParams, EnvCarry]):

    def __init__(self, maze_config: dict, name='keyroom'):
        super().__init__()
        self.maze_config = convert_dict_to_types(maze_config)
        self.task_objects, self.train_w, self.test_w = prepare_task_variables(
           self.maze_config)
        self.task_runner = TaskRunner(
           task_objects=self.task_objects,
           first_instance=True,
        )
        self.name = name

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        del params
        return spaces.Discrete(NUM_ACTIONS)


    def default_params(self, **kwargs) -> KeyRoomEnvParams:
        return KeyRoomEnvParams(height=19, width=19).replace(**kwargs)

    def time_limit(self, params: EnvParams) -> int:
        return 150

    def _generate_problem(self, params: KeyRoomEnvParams, rng: jax.Array) -> State[EnvCarry]:
        rng, *rngs = jax.random.split(rng, num=6)

        grid = nine_rooms(params.height, params.width)
        # grid = minigrid_common.prepare_grid(grid, extra_dims=2)
        roomW, roomH = params.width // 3, params.height // 3

        keys = self.maze_config['keys']
        pairs = self.maze_config['pairs']
        nrooms = len(keys)
        # assuming that rooms are square!
        if params.random_door_loc:
          door_coords = jax.random.randint(
              rngs[0], shape=(nrooms,), minval=1, maxval=roomW)
        else:
          door_coords = jnp.array([roomW//2]*nrooms)

        ###########################
        # helper functions
        ###########################
        def get_x_y(i,j):
          xL = i * roomW
          yT = j * roomH
          xR = xL + roomW
          yB = yT + roomH
          return xL, yT, xR, yB

        def sample_coordinates(i, j, rng, grid, off_border=True):
          xL, yT, xR, yB = get_x_y(i, j)
          width = lambda R, L: R - L
          height = lambda T, B: B - T

          if off_border:
              xL += 2
              yT += 2
              yB -= 1
              xR -= 1

          rng, rng_ = jax.random.split(rng)
          inner_coords = jax.random.choice(
              key=rng_,
              shape=(1,),
              a=jnp.arange(width(xR, xL) * height(yT, yB)),
              replace=False,
              p=free_tiles_mask(grid[xL:xR, yT:yB]).flatten(),
          )
          inner_coords = jnp.divmod(inner_coords, height(yT, yB))
          coords = (xL+inner_coords[0], yT+inner_coords[1])
          return coords, rng

        def place_in_room(i, j, rng, grid, obj: tuple, off_border=True):
          coords, rng = sample_coordinates(i, j, rng, grid)
          grid = grid.at[coords[0], coords[1]].set(obj)
          return grid, rng

        def add_top_door(grid, door):
          xL, yT, xR, yB = get_x_y(1, 0)
          return grid.at[yB, xL + door_coords[room_idx]].set(door)

        def add_right_door(grid, door):
          xL, yT, xR, yB = get_x_y(2, 1)
          return grid.at[yB - door_coords[room_idx], xL].set(door)

        def add_bottom_door(grid, door):
          xL, yT, xR, yB = get_x_y(1, 1)
          return grid.at[yB , xL + door_coords[room_idx]].set(door)

        def add_left_door(grid, door):
          xL, yT, xR, yB = get_x_y(1, 0)
          return grid.at[yB + door_coords[room_idx], xL].set(door)

        add_door_fns = [add_right_door, add_bottom_door,
                        add_left_door, add_top_door]

        ######################
        # add objects
        ######################
        all_obj_coords = [
            (1, 2),  # right
            (2, 1),  # bottom
            (1, 0),  # left
            (0, 1),  # top
        ]

        for room_idx in range(nrooms):

          #------------------
          # place key in room
          #------------------
          grid, rng = place_in_room(
              1, 1, rng, grid,
              make_obj(*keys[room_idx], visible=1))

          #------------------
          # add door
          #------------------
          add_door_fn = add_door_fns[room_idx]
          door_color = keys[room_idx][COLOR_IDX]
          door = make_obj(
             Tiles.DOOR_LOCKED, door_color,
             visible=1)
          grid = add_door_fn(grid, door)

          #------------------
          # add other objects
          #------------------
          obj_coords = all_obj_coords[room_idx]
          obj1, obj2 = pairs[room_idx]
          grid, rng = place_in_room(
              *obj_coords, rng, grid, make_obj(*obj1))

          grid, rng = place_in_room(
              *obj_coords, rng, grid, make_obj(*obj2))

        agent_position, rng = sample_coordinates(1, 1, rng, grid, off_border=False)

        agent = AgentState(
            position=jnp.concatenate(agent_position),
            direction=sample_direction(rngs[4]),
            pocket=make_obj_arr(Tiles.EMPTY, Colors.EMPTY),
            )

        goal_room_idx = jax.random.randint(rng, shape=(), minval=0, maxval=len(keys))
        goal_room = jax.nn.one_hot(goal_room_idx, len(keys))

        feature_weights = jax.lax.cond(
            params.training, 
            lambda: self.train_w*goal_room[:, None],
            lambda: self.test_w*goal_room[:, None],
        )

        state = EnvState(
            key=rng,
            step_num=jnp.asarray(0),
            grid=grid,
            agent=agent,
            local_agent_position=get_local_agent_position(
                agent.position, *grid.shape[:2]),
            room_grid=get_room_grid(grid=grid, agent=agent),
            goal_room_idx=goal_room_idx,
            feature_weights=feature_weights,
            carry=EnvCarry(),
        )
        return state

    # @partial(jax.jit, static_argnums=(0,))
    def reset(
       self, 
       key: jax.random.KeyArray,
       params: EnvParamsT) -> TimeStep[EnvCarryT]:
        state = self._generate_problem(params, key)

        task_state = self.task_runner.reset(
            visible_grid=state.room_grid,
            agent=state.agent)
        state = state.replace(task_state=task_state)

        observation = make_observation(state, state.room_grid)
        timestep = TimeStep(
            state=state,
            step_type=StepType.FIRST,
            reward=jnp.asarray(0.0),
            discount=jnp.asarray(1.0),
            observation=observation,
        )
        return timestep

    # @partial(jax.jit, static_argnums=(0,))
    def step(self,
             key: jax.random.KeyArray,
             timestep: TimeStep[EnvCarryT],
             action: IntOrArray,
             params: EnvParamsT,
             ) -> TimeStep[EnvCarryT]:
        del key
        new_grid, new_agent, _ = take_action(
            timestep.state.grid, timestep.state.agent, action)
        new_room_grid = get_room_grid(grid=new_grid, agent=new_agent)

        new_task_state = self.task_runner.step(
            prior_state=timestep.state.task_state,
            visible_grid=new_room_grid,
            agent=new_agent)
        new_state = timestep.state.replace(
            grid=new_grid,
            room_grid=new_room_grid,
            agent=new_agent,
            local_agent_position=get_local_agent_position(
                new_agent.position, *new_grid.shape[:2]),
            step_num=timestep.state.step_num + 1,
            task_state=new_task_state,
        )
        new_observation = make_observation(new_state, room_grid=new_room_grid)

        # checking for termination or truncation, choosing step type
        goal_room_objects = self.task_objects[new_state.goal_room_idx]

        def picked_up(task_object: jax.Array):
          pocket = make_task_obj(*new_agent.pocket, visible=1,
                                state=States.PICKED_UP, asarray=True)
          return (pocket == task_object).all()

        terminated = jax.lax.cond(
           params.training,
           # goal object picked up
           lambda: picked_up(goal_room_objects[TRAIN_OBJECT_IDX]),
           # goal key picked up
           lambda: picked_up(goal_room_objects[KEY_IDX]),
        )
        truncated = jnp.equal(new_state.step_num, self.time_limit(params))

        state_features = new_observation.state_features.astype(
            jnp.float32)

        reward = jax.lax.cond(
           params.training,
           # use accomplishment of state features as reward
           lambda: (state_features*new_observation.task_w).sum(),
           # was key for goal object picked up?
           lambda: picked_up(goal_room_objects[KEY_IDX]).astype(jnp.float32)
        )

        step_type = jax.lax.select(
            terminated | truncated, StepType.LAST, StepType.MID)

        discount = jax.lax.select(
            terminated, jnp.asarray(0.0), jnp.asarray(1.0))

        timestep = TimeStep(
            state=new_state,
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=new_observation,
        )
        return timestep
