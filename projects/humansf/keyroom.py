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

from xminigrid.core.constants import Colors, Tiles, NUM_ACTIONS, DIRECTIONS, TILES_REGISTRY
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

def index(x, i):
  return jax.lax.dynamic_index_in_dim(x, i, keepdims=False)

def accomplished(grid, task):
  # [H, W, D], [D]
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

def swap_test_pairs(maze_config: dict):
  maze_config = copy.deepcopy(maze_config)
  maze_config['pairs'][0][1], maze_config['pairs'][1][1] = maze_config['pairs'][1][1], maze_config['pairs'][0][1]
  try:
    maze_config['pairs'][2][1], maze_config['pairs'][3][1] = maze_config['pairs'][3][1], maze_config['pairs'][2][1]
  except:
     pass
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
    def add_visibility_pickup(grid):
        padding = jnp.array((1, 0))
        padding = jnp.tile(padding[None, None], (*grid.shape[:2], 1))
        return jnp.concatenate((grid, padding), axis=-1)

    # add dimension with all 1s, since each position here is visible to agent
    visible_grid = add_visibility_pickup(visible_grid)  

    # acc_in_grid is [num_tasks, num_task_objects]
    acc_in_grid = accomplished_HW(visible_grid, self.task_objects)

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
    random_door_loc: bool = struct.field(pytree_node=False, default=False)
    random_obj_loc: bool = struct.field(pytree_node=False, default=True)
    #train_single: bool = struct.field(pytree_node=False, default=True)
    train_multi_probs: float = struct.field(pytree_node=False, default=.5)
    training: bool = struct.field(pytree_node=False, default=True)
    time_limit: int = struct.field(pytree_node=False, default=150)
    maze_config: dict = None
    task_objects: jax.Array = None
    train_w: jax.Array = None
    test_w: jax.Array = None

class EnvState(struct.PyTreeNode, Generic[EnvCarryT]):
    key: jax.Array
    step_num: jax.Array

    grid: GridState
    room_grid: GridState
    agent: AgentState
    local_agent_position: jax.Array
    # task: Task
    carry: EnvCarryT
    room_setting: int
    task_w: jax.Array
    goal_room_idx: jax.Array
    task_object_idx: jax.Array
    offtask_w: Optional[jax.Array] = None
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
    room_y = agent_pos[0] // room_size
    room_x = agent_pos[1] // room_size

    # Calculate the starting and ending coordinates of the room
    start_x = room_x * room_size
    start_y = room_y * room_size

    delta = grid_width//3 + 1

    return jax.lax.dynamic_slice(
        grid, (start_y, start_x, 0), (delta, delta, grid_dim))

def get_local_agent_position(agent_pos, height, width):
    # Calculate the room size
    room_height = height // 3
    room_width = width // 3
    
    # Calculate the room indices
    room_y = agent_pos[0] // room_height
    room_x = agent_pos[1] // room_width
    
    # Calculate the starting coordinates of the room
    start_x = room_x * room_width
    start_y = room_y * room_height
    
    # Calculate the local position within the room
    local_y = agent_pos[0] - start_y
    local_x = agent_pos[1] - start_x

    return jnp.array([local_y, local_x])


###########################
# helper functions
###########################


def get_x_y(y, x, roomH, roomW):
  xL = x * roomW
  xR = xL + roomW

  yT = y * roomH
  yB = yT + roomH

  return xL, yT, xR, yB


def sample_coordinates(y, x, rng, grid, roomW=None, roomH=None, off_border=True):
  assert roomW is not None and roomH is not None
  xL, yT, xR, yB = get_x_y(y=y, x=x, roomW=roomW, roomH=roomH)

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
      a=jnp.arange(height(yT, yB) * width(xR, xL)),
      replace=False,
      p=free_tiles_mask(grid[yT:yB, xL:xR]).flatten(),
  )
  inner_coords = jnp.divmod(inner_coords, width(xR, xL))
  coords = (yT+inner_coords[0], xL+inner_coords[1])
  return coords, rng

def place_in_room(y, x, rng, grid, obj: tuple, off_border=True, roomW=None, roomH=None):
  assert roomW is not None and roomH is not None
  coords, rng = sample_coordinates(y=y, x=x, rng=rng, grid=grid, roomW=roomW, roomH=roomH)
  grid = grid.at[coords[0], coords[1]].set(obj)
  return grid, rng


def fixed_place_in_room(y, x, rng, grid, obj: tuple, idx: int, off_border=True, roomW=None, roomH=None):
  assert roomW is not None and roomH is not None
  xL, yT, xR, yB = get_x_y(y=y, x=x, roomW=roomW, roomH=roomH)
  x1 = xL + roomW//2 - 1  # 1 over from center of room
  x2 = xL + roomW//2 + 1  # 1 over from center of room
  y1 = yT + roomH//2  # center of room

  grid = grid.at[y1, x1 if idx else x2].set(obj)
  return grid, rng


class Observation(struct.PyTreeNode):
   image: jax.Array
   task_w: jax.Array
   state_features: jax.Array
   has_occurred: jax.Array
   pocket: jax.Array
   direction: jax.Array
   local_position: jax.Array
   position: jax.Array
   prev_action: jax.Array

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

def prepare_task_variables(
      maze_config: struct.PyTreeNode,
      key_reward: float = 0.25,
      door_reward: float = 5.0,
      ):
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
    train_w.append((key_reward, door_reward, 1., 0))
    test_w.append((0., 0., 0., 1.0))

  task_objects = jnp.array(task_objects)
  train_w = jnp.array(train_w)
  test_w = jnp.array(test_w)

  return task_objects, train_w, test_w

def make_observation(state: EnvState, prev_action: jax.Array, params: EnvParams):
  """This converts all inputs into binary vectors. this faciitates processing with a neural network."""

  binary_room_grid = minigrid_common.make_binary_vector_grid(state.room_grid)
  direction = jnp.zeros((minigrid_common.NUM_DIRECTIONS))
  direction = direction.at[state.agent.direction].set(1)

  local_position = minigrid_common.position_to_two_hot(
    state.local_agent_position, state.room_grid.shape[:2])

  global_position = minigrid_common.position_to_two_hot(
    state.agent.position, state.grid.shape[:2])

  observation = Observation(
      image=binary_room_grid,
      pocket=minigrid_common.make_binary_vector(state.agent.pocket),
      state_features=state.task_state.features.reshape(-1),
      has_occurred=(state.task_state.feature_counts > 0).reshape(-1),
      task_w=state.task_w.reshape(-1),
      direction=direction,
      local_position=local_position,
      position=global_position,
      prev_action=prev_action,
      )
  # just to be safe?
  observation = jax.tree_map(lambda x: jax.lax.stop_gradient(x), observation)
  return observation


def pair_object_picked_up(params, state):
    """True if any object in pairs is picked up."""
    pairs = params.maze_config['pairs'].reshape(-1, 2)
    pair_object_picked_up = (pairs == state.agent.pocket[None]).all(-1)
    return pair_object_picked_up.any()

class KeyRoom(Environment[KeyRoomEnvParams, EnvCarry]):

    def __init__(self,
                 train_episode_ends_on_pair_pickup: bool = True,
                 test_episode_ends_on: str = 'any_pair',
                 name='keyroom'):
        super().__init__()
        self.name = name
        self.train_episode_ends_on_pair_pickup = train_episode_ends_on_pair_pickup
        self.test_episode_ends_on = test_episode_ends_on

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        del params
        return spaces.Discrete(NUM_ACTIONS)

    def num_actions(self, params):
        return self.action_space(params).n

    def action_onehot(self, action, params):
        num_actions = self.num_actions(params) + 1
        one_hot = jnp.zeros((num_actions))
        one_hot = one_hot.at[action].set(1)
        return one_hot

    def default_params(
          self,
          maze_config: dict,
          height=19,
          width=19,
          **kwargs) -> KeyRoomEnvParams:

        maze_config = convert_dict_to_types(maze_config)

        task_objects, train_w, test_w = prepare_task_variables(
           maze_config)

        self.task_runner = TaskRunner(
           task_objects=task_objects,
           first_instance=True,
        )
        return KeyRoomEnvParams(height=height, width=width).replace(
          maze_config=maze_config,
          task_objects=task_objects,
          train_w=train_w,
          test_w=test_w,
          **kwargs)

    def time_limit(self, params: EnvParams) -> int:
        return params.time_limit

    def _generate_problem(self, params: KeyRoomEnvParams, rng: jax.Array) -> State[EnvCarry]:

      def single_or_multi(params_, rng_):
        """Training: either single room or multi-room"""
        rng_, rng2 = jax.random.split(rng_)
        train_multi = jax.random.bernoulli(rng2, params_.train_multi_probs)
        return jax.lax.cond(
          train_multi,
          lambda: self.multiroom_problem(params_, rng_),
          lambda: self.singleroom_problem(params_, rng_),
        )

      return jax.lax.cond(
         params.training,
         lambda: single_or_multi(params, rng),
         lambda: self.multiroom_problem(params, rng),
      )

    def singleroom_problem(self, params: KeyRoomEnvParams, rng: jax.Array) -> State[EnvCarry]:
        """For single room, pick final room. place both objects. select 1 as task object.

        Args:
            params (KeyRoomEnvParams): _description_
            rng (jax.Array): _description_

        Returns:
            State[EnvCarry]: _description_
        """
        #------------------
        # generate grid
        #------------------
        grid = nine_rooms(params.height, params.width)
        roomW, roomH = params.width // 3, params.height // 3


        #------------------
        # helper functions absorb room information
        #------------------
        place_in_room_p = partial(place_in_room, roomH=roomH, roomW=roomW)
        sample_coordinates_p = partial(sample_coordinates, roomH=roomH, roomW=roomW)

        #------------------
        # place objects in room
        #------------------
        pairs = params.maze_config['pairs']
        goal_room_idx = jax.random.randint(
            rng, shape=(), minval=0, maxval=len(pairs))
        pair = index(pairs, goal_room_idx)
        if params.random_obj_loc:
          grid, rng = place_in_room_p(
              1, 1, rng, grid,
              make_obj(*pair[0], visible=1))
          grid, rng = place_in_room_p(
              1, 1, rng, grid,
              make_obj(*pair[1], visible=1))
        else:
          grid, rng = fixed_place_in_room(
              1, 1, rng, grid,
              obj=make_obj(*pair[0], visible=1),
              idx=0,
              roomW=roomW,
              roomH=roomH,
              )
          grid, rng = fixed_place_in_room(
              1, 1, rng, grid,
              obj=make_obj(*pair[1], visible=1),
              idx=1,
              roomW=roomW,
              roomH=roomH,
              )

        #------------------
        # create agent
        #------------------
        if params.random_obj_loc:
          agent_position, rng = sample_coordinates_p(
              1, 1, rng, grid, off_border=False)
          agent_position = jnp.concatenate(agent_position)
        else:
          xL, yT, _, _ = get_x_y(y=1, x=1, roomW=roomW, roomH=roomH)
          agent_position = (yT+roomH//2, xL+roomW//2)
          agent_position = jnp.asarray(agent_position, dtype=jnp.int32)

        rng, rng_ = jax.random.split(rng)
        agent = AgentState(
            position=agent_position,
            direction=sample_direction(rng_),
            pocket=make_obj_arr(Tiles.EMPTY, Colors.EMPTY),
        )

        #------------------
        # define task
        #------------------
        goal_room = jax.nn.one_hot(goal_room_idx, len(pairs))
        # [num_rooms, task_dim]
        rng, rng_ = jax.random.split(rng)
        train_w = params.train_w*goal_room[:, None]
        test_w = params.test_w*goal_room[:, None]
        offtask_w = train_w
        task_w = test_w
        #task_w, offtask_w = jax.lax.cond(
        #    jax.random.bernoulli(rng_),
        #    lambda: (train_w, test_w),
        #    lambda: (test_w, train_w),
        #)

        state = EnvState(
            key=rng,
            step_num=jnp.asarray(0),
            grid=grid,
            agent=agent,
            local_agent_position=get_local_agent_position(
                agent.position, *grid.shape[:2]),
            room_grid=get_room_grid(grid=grid, agent=agent),
            room_setting=jnp.asarray(0, jnp.int32),
            goal_room_idx=goal_room_idx,
            task_object_idx=jnp.asarray(1, jnp.int32),
            task_w=task_w,
            offtask_w=offtask_w,
            carry=EnvCarry(),
        )
        return state

    def multiroom_problem(self, params: KeyRoomEnvParams, rng: jax.Array) -> State[EnvCarry]:
        rng, *rngs = jax.random.split(rng, num=6)

        grid = nine_rooms(params.height, params.width)
        # grid = minigrid_common.prepare_grid(grid, extra_dims=2)
        roomW, roomH = params.width // 3, params.height // 3

        keys = params.maze_config['keys']
        pairs = params.maze_config['pairs']
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
        get_x_y_p = partial(get_x_y, roomH=roomH, roomW=roomW)
        place_in_room_p = partial(place_in_room, roomH=roomH, roomW=roomW)
        sample_coordinates_p = partial(sample_coordinates, roomH=roomH, roomW=roomW)

        def add_right_door(grid, door):
          xL, yT, xR, yB = get_x_y_p(1, 2)
          return grid.at[yB - door_coords[room_idx], xL].set(door)

        def add_bottom_door(grid, door):
          xL, yT, xR, yB = get_x_y_p(1, 1)
          return grid.at[yB , xL + door_coords[room_idx]].set(door)

        def add_left_door(grid, door):
          xL, yT, xR, yB = get_x_y_p(0, 1)
          return grid.at[yB + door_coords[room_idx], xL].set(door)

        def add_top_door(grid, door):
          xL, yT, xR, yB = get_x_y_p(0, 1)
          return grid.at[yB, xL + door_coords[room_idx]].set(door)

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

        all_key_pos = [
           (3, 4),
           (4, 3),
           (3, 2),
           (2, 3),
        ]
        for room_idx in range(nrooms):

          #------------------
          # place key in room
          #------------------
          if params.random_obj_loc:
            grid, rng = place_in_room_p(
                1, 1, rng, grid,
                make_obj(*keys[room_idx], visible=1))
          else:
            xL, yT, _, _ = get_x_y(y=1, x=1, roomW=roomW, roomH=roomH)
            y, x = all_key_pos[room_idx]
            grid = grid.at[yT+y, xL+x].set(make_obj(*keys[room_idx], visible=1))

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
          if params.random_obj_loc:
            grid, rng = place_in_room_p(
                *obj_coords, rng, grid, make_obj(*obj1))

            grid, rng = place_in_room_p(
                *obj_coords, rng, grid, make_obj(*obj2))
          else:
            grid, rng = fixed_place_in_room(
                *obj_coords, rng, grid,
                make_obj(*obj1, visible=1),
                idx=0,
                roomW=roomW,
                roomH=roomH,
            )
            grid, rng = fixed_place_in_room(
                *obj_coords, rng, grid,
                make_obj(*obj2, visible=1),
                idx=1,
                roomW=roomW,
                roomH=roomH
            )

        #------------------
        # create agent
        #------------------
        if params.random_obj_loc:
          agent_position, rng = sample_coordinates_p(1, 1, rng, grid, off_border=False)
          agent_position = jnp.concatenate(agent_position)
        else:
          xL, yT, _, _ = get_x_y(y=1, x=1, roomW=roomW, roomH=roomH)
          agent_position = (yT+3, xL+3)
          agent_position = jnp.asarray(agent_position, dtype=jnp.int32)

        agent = AgentState(
            position=agent_position,
            direction=sample_direction(rngs[4]),
            pocket=make_obj_arr(Tiles.EMPTY, Colors.EMPTY),
            )

        #------------------
        # define task
        #------------------
        goal_room_idx = jax.random.randint(rng, shape=(), minval=0, maxval=len(keys))
        goal_room = jax.nn.one_hot(goal_room_idx, len(keys))

        def get_train_object(rng):
           feature_weights = params.train_w*goal_room[:, None]
           task_object_idx = 0
           return feature_weights, task_object_idx

        def get_test_object(rng):
           feature_weights = params.test_w*goal_room[:, None]
           task_object_idx = 1
           return feature_weights, task_object_idx

        def get_train_or_test(rng):
           get_train = jax.random.bernoulli(rng)
           return jax.lax.cond(
            get_train,
            get_train_object,
            get_test_object,
            rng
            )

        rng, rng_ = jax.random.split(rng)
        feature_weights, task_object_idx = jax.lax.cond(
          params.training,
          get_train_object,
          get_train_or_test,
          rng_,
        )
        offtask_w = params.test_w*goal_room[:, None]

        state = EnvState(
            key=rng,
            step_num=jnp.asarray(0),
            grid=grid,
            agent=agent,
            local_agent_position=get_local_agent_position(
                agent.position, *grid.shape[:2]),
            room_grid=get_room_grid(grid=grid, agent=agent),
            task_object_idx=task_object_idx,
            room_setting=jnp.asarray(1, jnp.int32),
            goal_room_idx=goal_room_idx,
            task_w=feature_weights,
            offtask_w=offtask_w,
            carry=EnvCarry(),
        )
        return state

    def single_room_reward_termination(self, params, state: EnvState, observation: Observation):

      def reward_fn(state: EnvState, observation: Observation):
        state_features = observation.state_features.astype(
          jnp.float32)
        return (state_features*observation.task_w).sum()

      def termination_fn(state: EnvState, observation: Observation):
        pocket = state.agent.pocket
        train_ends_on_task_pickup = False
        if train_ends_on_task_pickup:
          # terminate when task object has been picked up
          goal_room_objects = params.task_objects[state.goal_room_idx]
          task_object = goal_room_objects[TEST_OBJECT_IDX]
          return (pocket == task_object).all()
        else:
          # terminate when __anything__ has been picked up
          return pocket[0] != Tiles.EMPTY

      reward = reward_fn(state, observation)
      termination = termination_fn(state, observation)
      return reward, termination

    def multi_room_reward_termination(self, params, state: EnvState, observation: Observation):

      def reward_fn_train(s: EnvState, o: Observation):
        state_features = o.state_features.astype(
          jnp.float32)
        return (state_features*o.task_w).sum()

      def reward_fn_test(s: EnvState, o: Observation):
        if self.test_episode_ends_on == 'any_pair':
            return reward_fn_train(s, o)
        elif self.test_episode_ends_on == 'any_key':
            # rewarded if pick up correct key
            pocket = s.agent.pocket
            goal_room_objects = index(params.task_objects, s.goal_room_idx)
            picked_up = (pocket == goal_room_objects[KEY_IDX][:2]).all()
            return picked_up.astype(jnp.float32)
        else:
            raise NotImplementedError

      def termination_fn_train(s: EnvState, o: Observation):
        if self.train_episode_ends_on_pair_pickup:
            return pair_object_picked_up(params, s)
        else:
            goal_room_objects = index(params.task_objects, s.goal_room_idx)
            task_object = goal_room_objects[TRAIN_OBJECT_IDX]
            return (s.agent.pocket == task_object[:2]).all()

      def termination_fn_test(s: EnvState, o: Observation):
        if self.test_episode_ends_on == 'any_pair':
            return pair_object_picked_up(params, s)
        elif self.test_episode_ends_on == 'any_key':
            # terminate when agent picks up key.
            pocket = s.agent.pocket
            return pocket[0] == Tiles.KEY

      def reward_fn(state: EnvState, observation: Observation):
        return jax.lax.cond(
          params.training,
          reward_fn_train,
          reward_fn_test,
          state, observation
        )

      def termination_fn(state: EnvState, observation: Observation):
        return jax.lax.cond(
          params.training,
          termination_fn_train,
          termination_fn_test,
          state, observation
        )

      reward = reward_fn(state, observation)
      termination = termination_fn(state, observation)
      return reward, termination

    @partial(jax.jit, static_argnums=(0,))
    def reset(
       self, 
       key: jax.random.PRNGKey,
       params: EnvParamsT) -> TimeStep[EnvCarryT]:
        state = self._generate_problem(params, key)

        task_state = self.task_runner.reset(
            visible_grid=state.room_grid,
            agent=state.agent)
        state = state.replace(task_state=task_state)

        observation = make_observation(
           state,
           prev_action=self.action_onehot(self.num_actions(params), params),
           params=params)

        timestep = TimeStep(
            state=state,
            step_type=StepType.FIRST,
            reward=jnp.asarray(0.0),
            discount=jnp.asarray(1.0),
            observation=observation,
        )
        return timestep

    def teleport_agent_remove_key_close_door(
          self,
          prior_timestep,
          new_grid,
          new_agent):
        def move(position: jax.Array,
                 direction: jax.Array,
                 steps: int = 2) -> jax.Array:
          direction = index(DIRECTIONS, direction)
          new_position = position + direction*steps
          return new_position

        def teleport_agent_remove_key_close_door(grid, agent, door_opened_somewhere):
          # get door identity
          y, x = jnp.nonzero(
              door_opened_somewhere, size=1)
          door = grid[y[0], x[0]]  # [D=2]

          # get which task object door corresponded to
          # [num_tasks, num_task_objects]
          which_task_obj_is_door = (
              door[None, None] == self.task_runner.task_objects[:, :, :2]).all(-1)

          # [num_tasks, num_task_objects]
          feature_counts = prior_timestep.state.task_state.feature_counts
          feature_counts_door = (which_task_obj_is_door.astype(
              jnp.float32)*feature_counts).sum()

          def update_agent_grid(g, a):
            position = move(a.position, a.direction)
            new_a = a.replace(
                position=position,
                pocket=TILES_REGISTRY[Tiles.EMPTY, Colors.EMPTY])

            new_g = g.at[y[0], x[0]].set(
                TILES_REGISTRY[Tiles.DOOR_CLOSED, door[COLOR_IDX]])

            return new_g, new_a

          new_grid, new_agent = jax.lax.cond(
              feature_counts_door < 1,
              update_agent_grid,
              lambda g, a: (g, a),
              grid, agent,
          )
          return new_grid, new_agent

        # --------
        # door opened?
        # --------
        door_opened_somewhere = (
            Tiles.DOOR_OPEN == new_grid[:, :, 0])
        door_opened = door_opened_somewhere.any()

        return jax.lax.cond(
            door_opened,
            lambda: teleport_agent_remove_key_close_door(
                new_grid, new_agent, door_opened_somewhere),
            lambda: (new_grid, new_agent),
        )

    def take_action(self,
             key: jax.random.PRNGKey,
             timestep: TimeStep[EnvCarryT],
             action: IntOrArray,
             params: EnvParamsT,
             ) -> TimeStep[EnvCarryT]:
        del key
        del params
        new_grid, new_agent, _ = take_action(
            timestep.state.grid, timestep.state.agent, action)

        new_grid, new_agent = self.teleport_agent_remove_key_close_door(
            prior_timestep=timestep,
            new_grid=new_grid,
            new_agent=new_agent)

        return new_grid, new_agent, _

    @partial(jax.jit, static_argnums=(0,))
    def step(self,
             key: jax.random.PRNGKey,
             timestep: TimeStep[EnvCarryT],
             action: IntOrArray,
             params: EnvParamsT,
             ) -> TimeStep[EnvCarryT]:
        new_grid, new_agent, _ = self.take_action(
            key=key, timestep=timestep, action=action, params=params)
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
        new_observation = make_observation(
           new_state,
           prev_action=self.action_onehot(action, params),
           params=params)

        reward, terminated = jax.lax.cond(
            new_state.room_setting==jnp.asarray(0, jnp.int32),
            self.single_room_reward_termination,
            self.multi_room_reward_termination,
            params, new_state, new_observation,
        )

        truncated = jnp.equal(new_state.step_num, self.time_limit(params))

        step_type = jax.lax.select(
            terminated | truncated, StepType.LAST, StepType.MID)

        discount = jax.lax.select(
            terminated, jnp.asarray(0.0), jnp.asarray(1.0))

        new_timestep = TimeStep(
            state=new_state,
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=new_observation,
        )
        return new_timestep
