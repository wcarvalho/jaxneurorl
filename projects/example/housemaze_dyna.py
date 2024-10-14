from typing import Optional, List, Callable, Tuple

import random

import numpy as np
import jax
import jax.tree_util as jtu
import jax.numpy as jnp
from flax import struct
import distrax
import pdb

from housemaze import levels
from housemaze import env as maze
from housemaze import renderer
from housemaze import utils

import itertools

maze1 = """
.....
..>..
.....
A....
.....
""".strip()

maze2 = """
..B..
.....
.....
...<.
.....
""".strip()

maze3 = """
.....
.....
..C..
.....
.>...
""".strip()

maze4 = """
...>.
.....
.....
.....
.D...
""".strip()

maze5 = """
.....
.....
...<E
.....
.....
""".strip()

def generate_test_maze(objects, n_test_obj = 2):
    # Define a blank 5x5 maze
    maze = [['.' for _ in range(5)] for _ in range(5)]

    # Choose two random objects from A, B, C, D, E
    objects = random.sample(objects, n_test_obj)

    # Randomly select three unique positions in the grid (for two objects and one agent)
    positions = random.sample([(r, c) for r in range(5) for c in range(5)], n_test_obj + 1)

    # Place the objects at the first two positions
    maze[positions[0][0]][positions[0][1]] = objects[0]
    maze[positions[1][0]][positions[1][1]] = objects[1]

    # Place the agent at the third position with a random direction
    direction = random.choice(['<', '>'])
    maze[positions[2][0]][positions[2][1]] = direction

    # Convert the maze list back to a string for display
    return "\n".join("".join(row) for row in maze)

@struct.dataclass
class EnvParamsTask:
    map_init: maze.MapInit
    objects: jax.Array
    time_limit: int = 100
    n_train: int = 5

@struct.dataclass
class EnvStateTask:
    # episode information
    key: jax.Array
    step_num: jax.Array

    # map info
    grid: jax.Array
    agent_pos: jax.Array
    agent_dir: int

    # task info
    map_idx: jax.Array
    task_w: jax.Array
    train_vector: jax.Array
    task_state: Optional[maze.TaskState] = None
    is_train: int = 0

class ObservationTask(struct.PyTreeNode):
   image: jax.Array
   task_w: jax.Array
   state_features: jax.Array
   position: jax.Array
   direction: jax.Array
   prev_action: jax.Array
   is_train: jax.Array
   map_idx: jax.Array
   train_vector: jax.Array

class HouseMazeTasks(maze.HouseMaze):
    def make_observation(
        self,
        state: EnvStateTask,
        prev_action: jax.Array):
        """This converts all inputs into categoricals.

        Categories are [objects, directions, spatial positions, actions]
        """
        grid = state.grid
        agent_pos = state.agent_pos
        agent_dir = state.agent_dir

        # Compute the total number of categories
        num_object_categories = self.num_categories
        num_directions = len(maze.DIR_TO_VEC)
        H, W = grid.shape[-3:-1]
        num_spatial_positions = H*W

        # Convert direction to the right category integer. after [objects]
        start = num_object_categories
        direction_category = start + agent_dir

        # Convert position to the right category integer. after [objects, directions]
        start = num_object_categories + num_directions
        position_category = (
            start + agent_pos[0],
            start + H + agent_pos[1])
        # Convert prev_action to the right category integer. after [objects, directions, spatial positions]
        start = num_object_categories + num_directions + H + W
        prev_action_category = start + prev_action

        observation = ObservationTask(
            image=jnp.squeeze(state.grid).astype(jnp.int32),
            state_features=state.task_state.features.astype(jnp.float32),
            task_w=state.task_w.astype(jnp.float32),
            direction=jnp.array(direction_category, dtype=jnp.int32),
            position=jnp.array(position_category, dtype=jnp.int32),
            prev_action=jnp.array(prev_action_category, dtype=jnp.int32),
            is_train=jnp.array(state.is_train, dtype=jnp.int32),
            map_idx=jnp.array(state.map_idx, dtype=jnp.int32),
            train_vector=jnp.array(state.train_vector, dtype=jnp.int32),
        )

        # Just to be safe?
        observation = jax.tree_map(lambda x: jax.lax.stop_gradient(x), observation)
        return observation


    def reset(self, rng: jax.Array, params: EnvParamsTask) -> maze.TimeStep:
        """
        Sample map and then sample random object in map as task object.
        """
        ndim = params.map_init.grid.ndim
        if ndim == 3:
            # single choice
            map_idx = jnp.array(0)
            map_init = params.map_init
        elif ndim == 4:
            # multiple to choose from
            nlevels = len(params.map_init.grid)
            rng, rng_ = jax.random.split(rng)

            # select one
            map_idx = jax.random.randint(
                rng_, shape=(), minval=0, maxval=nlevels)

            # index across each pytree member
            def index(p): return jax.lax.dynamic_index_in_dim(
                p, map_idx, keepdims=False)
            map_init = jax.tree_map(index, params.map_init)
        else:
            raise NotImplementedError

        grid = map_init.grid
        agent_dir = map_init.agent_dir
        agent_pos = map_init.agent_pos

        ##################
        # sample task object
        ##################
        present_objects = (grid == params.objects[None, None])
        present_objects = present_objects.any(axis=(0,1))
        #jax.debug.print("{x}", x=present_objects)
        object_sampler = distrax.Categorical(
            logits=present_objects.astype(jnp.float32))
        rng, rng_ = jax.random.split(rng)
        object_idx = object_sampler.sample(seed=rng_)
        task_object = jax.lax.dynamic_index_in_dim(
            params.objects, object_idx, keepdims=False,
        )

        ##################
        # create task vectors
        ##################
        task_w = self.task_runner.task_vector(task_object)
        task_state = self.task_runner.reset(grid, agent_pos)

        ##################
        # create ouputs
        ##################
        vector = jnp.zeros(params.n_train)
        state = EnvStateTask(
            key=rng,
            step_num=jnp.asarray(0),
            grid=grid,
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            map_idx=map_idx,
            task_w=task_w,
            task_state=task_state,
            is_train=jax.lax.cond(map_idx < params.n_train, lambda: 1, lambda: 0),
            train_vector=jax.lax.cond(map_idx >= params.n_train,
                          lambda _: jnp.ones(params.n_train),            # If X >= D, all ones
                          lambda _: vector.at[map_idx].set(1),    # Else set 1 at index X
                          operand=None)
        )

        reset_action = jnp.array(self.num_actions() + 1, dtype=jnp.int32)
        timestep = maze.TimeStep(
            state=state,
            step_type=maze.StepType.FIRST,
            reward=jnp.asarray(0.0),
            discount=jnp.asarray(1.0),
            observation=self.make_observation(
                state,
                prev_action=reset_action)
        )
        pdb.set_trace()
        return timestep

    def step(self, rng: jax.Array, timestep: maze.TimeStep, action: jax.Array, params: EnvParamsTask) -> maze.TimeStep:

        if self.action_spec == 'keyboard':
            grid, agent_pos, agent_dir = maze.take_action(
                timestep.state.replace(agent_dir=action),
                action=maze.MinigridActions.forward)
        elif self.action_spec == 'minigrid':
            grid, agent_pos, agent_dir = maze.take_action(
                timestep.state,
                action)
        else:
            raise NotImplementedError(self.action_spec)

        task_state = self.task_runner.step(
            timestep.state.task_state, grid, agent_pos)

        state = timestep.state.replace(
            grid=grid,
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            task_state=task_state,
            step_num=timestep.state.step_num + 1,
        )

        task_w = timestep.state.task_w.astype(jnp.float32)
        features = task_state.features.astype(jnp.float32)
        reward = (task_w*features).sum(-1)
        terminated = self.task_runner.check_terminated(features, task_w)
        truncated = jnp.equal(state.step_num, params.time_limit)

        step_type = jax.lax.select(
            terminated | truncated, maze.StepType.LAST, maze.StepType.MID)
        discount = jax.lax.select(
            terminated, jnp.asarray(0.0), jnp.asarray(1.0))

        timestep = maze.TimeStep(
            state=state,
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=self.make_observation(
                state,
                prev_action=action),
        )
        pdb.set_trace()
        return timestep


def get_random_test_task():
    # Define the number of tasks (one-hot encoded)
    num_tasks = 3  # Adjust if needed

    # Create all possible combinations where a+b+c = 2 (i.e., exactly two tasks are selected)
    combinations = list(itertools.combinations(range(num_tasks), 2))

    # Generate new tasks as linear combinations
    task_list = []
    for comb in combinations:
        # Create a one-hot vector for each combination
        task_vector = np.zeros(num_tasks)

        # Set the selected tasks to 1
        for index in comb:
            task_vector[index] = 1

        task_list.append(task_vector)
    return task_list

def housemaze_env():
    char_to_key=dict(
        A="knife",
        B="fork",
        C="pan",
        D="pot",
        E="bowl",
        F="plates",
    )

    train_mazes = [maze1, maze2, maze3, maze4, maze5]
    test_mazes = [generate_test_maze(list(char_to_key.keys()), 2) for _ in range(5)] # 5 test tasks
    mazes = train_mazes + test_mazes

    image_dict = utils.load_image_dict()

    object_to_index = {key: idx for idx, key in enumerate(image_dict['keys'])}

    objects = np.array([object_to_index[v] for v in char_to_key.values()])

    map_inits = [utils.from_str(maze,
                                char_to_key=char_to_key,
                                object_to_index=object_to_index)
                 for maze in mazes]

    map_init = jtu.tree_map(lambda *v: jnp.stack(v), *map_inits)
    map_init = maze.MapInit(*map_init)
    # create env params
    env_params = EnvParamsTask(
        map_init=jax.tree_map(jnp.asarray, map_init),
        time_limit=jnp.array(50),
        objects=jnp.asarray(objects),
        n_train=len(train_mazes))

    seed = 6
    rng = jax.random.PRNGKey(seed)

    task_runner = maze.TaskRunner(task_objects=env_params.objects)
    env = HouseMazeTasks(
        task_runner=task_runner,
        num_categories=len(image_dict['keys']),
    )
    env = utils.AutoResetWrapper(env)

    return env_params, env