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
    #train_single: bool = struct.field(pytree_node=False, default=True)
    train_multi_probs: float = struct.field(pytree_node=False, default=.5)
    training: bool = struct.field(pytree_node=False, default=True)
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
    feature_weights: jax.Array
    goal_room_idx: jax.Array
    task_object_idx: jax.Array
    termination_object: jax.Array
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

class Observation(struct.PyTreeNode):
   image: jax.Array
   task_w: jax.Array
   state_features: jax.Array
   has_occurred: jax.Array
   pocket: jax.Array
   direction: jax.Array
   local_position: jax.Array
   position: jax.Array

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
    train_w.append((.25, .5, 1., 0))
    test_w.append((0., 0., 0., 1.0))

  task_objects = jnp.array(task_objects)
  train_w = jnp.array(train_w)
  test_w = jnp.array(test_w)

  return task_objects, train_w, test_w

def make_observation(state: EnvState):
  """This converts all inputs into binary vectors. this faciitates processing with a neural network."""

  binary_room_grid = minigrid_common.make_binary_vector_grid(state.room_grid)
  direction = jnp.zeros((minigrid_common.NUM_DIRECTIONS))
  direction.at[state.agent.direction].set(1)

  local_position = minigrid_common.position_to_two_hot(
    state.local_agent_position, state.room_grid.shape[:2])

  global_position = minigrid_common.position_to_two_hot(
    state.agent.position, state.grid.shape[:2])

  observation = Observation(
      image=binary_room_grid,
      pocket=minigrid_common.make_binary_vector(state.agent.pocket),
      state_features=state.task_state.features.reshape(-1),
      has_occurred=(state.task_state.feature_counts > 0).reshape(-1),
      task_w=state.feature_weights.reshape(-1),
      direction=direction,
      local_position=local_position,
      position=global_position,
      )
  # just to be safe?
  observation = jax.tree_map(lambda x: jax.lax.stop_gradient(x), observation)
  return observation

class KeyRoom(Environment[KeyRoomEnvParams, EnvCarry]):

    def __init__(self, test_end_on_key: bool = True, name='keyroom'):
        super().__init__()
        self.name = name
        self.test_end_on_key = test_end_on_key

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        del params
        return spaces.Discrete(NUM_ACTIONS)

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
        return 150

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
        # place test objects in room
        #------------------
        pairs = params.maze_config['pairs']
        for _, test_obj in pairs:
          grid, rng = place_in_room_p(
              1, 1, rng, grid,
              make_obj(*test_obj, visible=1))

        #------------------
        # create agent
        #------------------
        agent_position, rng = sample_coordinates_p(
            1, 1, rng, grid, off_border=False)
        rng, rng_ = jax.random.split(rng)
        agent = AgentState(
            position=jnp.concatenate(agent_position),
            direction=sample_direction(rng_),
            pocket=make_obj_arr(Tiles.EMPTY, Colors.EMPTY),
        )

        #------------------
        # define task
        #------------------
        goal_room_idx = jax.random.randint(
            rng, shape=(), minval=0, maxval=len(pairs))
        goal_room = jax.nn.one_hot(goal_room_idx, len(pairs))
        goal_room_objects = params.task_objects[goal_room_idx]

        # w-vector
        feature_weights = params.test_w*goal_room[:, None]

        # define object which specifies whether 
        termination_object = goal_room_objects[TEST_OBJECT_IDX]

        state = EnvState(
            key=rng,
            step_num=jnp.asarray(0),
            grid=grid,
            agent=agent,
            local_agent_position=get_local_agent_position(
                agent.position, *grid.shape[:2]),
            room_grid=get_room_grid(grid=grid, agent=agent),
            room_setting=0,
            goal_room_idx=goal_room_idx,
            task_object_idx=1,
            feature_weights=feature_weights,
            termination_object=termination_object,
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

        for room_idx in range(nrooms):

          #------------------
          # place key in room
          #------------------
          grid, rng = place_in_room_p(
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
          grid, rng = place_in_room_p(
              *obj_coords, rng, grid, make_obj(*obj1))

          grid, rng = place_in_room_p(
              *obj_coords, rng, grid, make_obj(*obj2))

        #------------------
        # create agent
        #------------------
        agent_position, rng = sample_coordinates_p(1, 1, rng, grid, off_border=False)
        agent = AgentState(
            position=jnp.concatenate(agent_position),
            direction=sample_direction(rngs[4]),
            pocket=make_obj_arr(Tiles.EMPTY, Colors.EMPTY),
            )

        #------------------
        # define task
        #------------------
        goal_room_idx = jax.random.randint(rng, shape=(), minval=0, maxval=len(keys))
        goal_room = jax.nn.one_hot(goal_room_idx, len(keys))
        goal_room_objects = params.task_objects[goal_room_idx]

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

        test_end_idx = KEY_IDX if self.test_end_on_key else 2+task_object_idx
        termination_object = jnp.where(
            params.training,
            # goal object picked up
            goal_room_objects[TRAIN_OBJECT_IDX],
            # goal key picked up
            goal_room_objects[test_end_idx],
        )
        state = EnvState(
            key=rng,
            step_num=jnp.asarray(0),
            grid=grid,
            agent=agent,
            local_agent_position=get_local_agent_position(
                agent.position, *grid.shape[:2]),
            room_grid=get_room_grid(grid=grid, agent=agent),
            task_object_idx=task_object_idx,
            room_setting=1,
            goal_room_idx=goal_room_idx,
            feature_weights=feature_weights,
            termination_object=termination_object,
            carry=EnvCarry(),
        )
        return state

    @partial(jax.jit, static_argnums=(0,))
    def reset(
       self, 
       key: jax.random.KeyArray,
       params: EnvParamsT) -> TimeStep[EnvCarryT]:
        state = self._generate_problem(params, key)

        task_state = self.task_runner.reset(
            visible_grid=state.room_grid,
            agent=state.agent)
        state = state.replace(task_state=task_state)

        observation = make_observation(state)
        timestep = TimeStep(
            state=state,
            step_type=StepType.FIRST,
            reward=jnp.asarray(0.0),
            discount=jnp.asarray(1.0),
            observation=observation,
        )
        return timestep

    def take_action(self,
             key: jax.random.KeyArray,
             timestep: TimeStep[EnvCarryT],
             action: IntOrArray,
             params: EnvParamsT,
             ) -> TimeStep[EnvCarryT]:
        del key
        del params
        return take_action(
            timestep.state.grid, timestep.state.agent, action)


    @partial(jax.jit, static_argnums=(0,))
    def step(self,
             key: jax.random.KeyArray,
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
        new_observation = make_observation(new_state)

        # checking for termination or truncation
        def picked_up(task_object: jax.Array):
          pocket = make_task_obj(*new_agent.pocket, visible=1,
                                state=States.PICKED_UP, asarray=True)
          return (pocket == task_object).all()

        terminated = picked_up(new_state.termination_object)
        truncated = jnp.equal(new_state.step_num, self.time_limit(params))

        state_features = new_observation.state_features.astype(
            jnp.float32)
        goal_room_objects = params.task_objects[new_state.goal_room_idx]

        reward = jax.lax.cond(
           params.training,
           # use accomplishment of state features as reward
           lambda: (state_features*new_observation.task_w).sum(),
           # was key for goal object picked up?
           lambda: picked_up(goal_room_objects[KEY_IDX]).astype(jnp.float32) if self.test_end_on_key else (state_features*new_observation.task_w).sum()
        )

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
