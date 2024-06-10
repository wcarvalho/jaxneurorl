
from typing import Optional
import jax
import jax.numpy as jnp
from flax import struct
import distrax

from projects.humansf.housemaze import maze

TaskRunner = maze.TaskRunner
TimeStep = maze.TimeStep
StepType = maze.StepType

MapInit = maze.MapInit


@struct.dataclass
class ResetParams:
    map_init: maze.MapInit
    train_objects: jax.Array
    test_objects: jax.Array
    starting_locs: Optional[jax.Array] = None
    curriculum: jax.Array = jnp.array(False)


@struct.dataclass
class EnvParams:
    reset_params: ResetParams
    time_limit: int = 250
    training: bool = True


@struct.dataclass
class EnvState:
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
    is_train_task: jax.Array
    task_object: jax.Array
    offtask_w: jax.Array
    task_state: Optional[maze.TaskState] = None


def mask_sample(mask, rng):
    # Creating logits based on the mask: -1e8 where mask is 0, 1 where mask is 1
    logits = jnp.where(mask == 1, mask.astype(jnp.float32), -1e8).astype(jnp.float32)

    # Creating the Categorical distribution with the specified logits
    sampler = distrax.Categorical(logits=logits)

    # Splitting the RNG
    rng, rng_ = jax.random.split(rng)

    # Sampling from the distribution
    return sampler.sample(seed=rng_)

class HouseMaze(maze.HouseMaze):

    def reset(self, rng: jax.Array, params: EnvParams) -> TimeStep:
        """
        
        1. Sample level.
        """
        ##################
        # sample level
        ##################
        nlevels = len(params.reset_params.curriculum)
        rng, rng_ = jax.random.split(rng)
        reset_params_idx = jax.random.randint(
            rng_, shape=(), minval=0, maxval=nlevels)

        def index(p): return jax.lax.dynamic_index_in_dim(
            p, reset_params_idx, keepdims=False)
        reset_params = jax.tree_map(index, params.reset_params)

        grid = reset_params.map_init.grid
        agent_dir = reset_params.map_init.agent_dir

        ##################
        # sample pair
        ##################
        pair_idx = mask_sample(mask=reset_params.train_objects >= 0, rng=rng)

        ##################
        # sample position (function of which pair has been choice)
        ##################
        def sample_pos_from_curriculum(rng_):
            locs = jax.lax.dynamic_index_in_dim(
                reset_params.starting_locs, pair_idx, keepdims=False)
            loc_idx = mask_sample(mask=(locs >= 0).all(-1), rng=rng)
            loc = jax.lax.dynamic_index_in_dim(
                locs, loc_idx, keepdims=False)
            return loc

        rng, rng_ = jax.random.split(rng)
        agent_pos = jax.lax.cond(
            jnp.logical_and(reset_params.curriculum, params.training),
            sample_pos_from_curriculum,
            lambda _: reset_params.map_init.agent_pos,
            rng_
        )

        ##################
        # sample task objects
        ##################
        train_object = jax.lax.dynamic_index_in_dim(
            reset_params.train_objects, pair_idx, keepdims=False,
        )
        test_object = jax.lax.dynamic_index_in_dim(
            reset_params.test_objects, pair_idx, keepdims=False,
        )

        def train_sample(rng):
            is_train_task = jnp.array(True)
            return train_object, test_object, is_train_task

        def test_sample(rng):
            is_train_task = jnp.array(False)
            return test_object, train_object, is_train_task

        def train_or_test_sample(rng):
            return jax.lax.cond(
                jax.random.bernoulli(rng),
                train_sample,
                test_sample,
                rng
            )
        rng, rng_ = jax.random.split(rng)
        task_object, offtask_object, is_train_task = jax.lax.cond(
            params.training,
            train_sample,
            train_or_test_sample,
            rng_
        )

        ##################
        # create task vectors
        ##################

        task_w = self.task_runner.task_vector(task_object)
        offtask_w = self.task_runner.task_vector(offtask_object)
        task_state = self.task_runner.reset(grid, agent_pos)

        ##################
        # create ouputs
        ##################
        state = EnvState(
            key=rng,
            step_num=jnp.asarray(0),
            grid=grid,
            is_train_task=is_train_task,
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            map_idx=reset_params_idx,
            task_w=task_w,
            task_object=task_object,
            offtask_w=offtask_w,
            task_state=task_state,
        )

        reset_action = self.num_actions(None) + 1
        timestep = TimeStep(
            state=state,
            step_type=StepType.FIRST,
            reward=jnp.asarray(0.0),
            discount=jnp.asarray(1.0),
            observation=self.make_observation(
                state,
                prev_action=reset_action)
        )
        return timestep

    def step(self, rng: jax.Array, timestep: TimeStep, action: jax.Array, params: EnvParams) -> TimeStep:

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
        )

        terminated = (task_state.features > 0).any()  # any object picked up
        task_w = timestep.state.task_w.astype(jnp.float32)
        features = task_state.features.astype(jnp.float32)
        reward = (task_w*features).sum(-1)
        truncated = jnp.equal(state.step_num, params.time_limit)

        step_type = jax.lax.select(
            terminated | truncated, StepType.LAST, StepType.MID)
        discount = jax.lax.select(
            terminated, jnp.asarray(0.0), jnp.asarray(1.0))

        timestep = TimeStep(
            state=state,
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=self.make_observation(
                state,
                prev_action=action),
        )

        return timestep
