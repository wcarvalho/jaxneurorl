
import os
import webrl

from typing import NamedTuple
import numpy as np
import jax.numpy as jnp


from webrl import jax_utils

from housemaze.human_dyna import env as maze
from housemaze.human_dyna import utils
from housemaze.human_dyna import mazes


from dotenv import load_dotenv

load_dotenv()
DEBUG_APP = int(os.environ.get('DEBUG_APP', 0))
DEBUG_SEED = os.environ.get('DEBUG_SEED', 1)

############
# Set up environment
############

list_of_groups = utils.load_groups()


def make_env_params(maze_str: str, group_set):
    return mazes.make_env_params(
        maze_str=maze_str,
        group_set=group_set,
    ).replace(
        training=False,
        terminate_with_done=2,
    )


practice_env_params = make_env_params(mazes.maze0, list_of_groups[0])
env1_params = make_env_params(mazes.maze1, list_of_groups[1])
env2_params = make_env_params(mazes.maze2, list_of_groups[2])
env3_params = make_env_params(mazes.maze3, list_of_groups[3])
env4_params = make_env_params(mazes.maze4, list_of_groups[4])


image_data = utils.load_image_dict()
json_image_data = jax_utils.convert_to_serializable(image_data)

task_objects = list_of_groups[:5].reshape(-1)
task_objects = jnp.unique(task_objects)
task_runner = maze.TaskRunner(
    task_objects=task_objects)
keys = image_data['keys']

jax_env = maze.HouseMaze(
    task_runner=task_runner,
    num_categories=len(keys),
    use_done=True,
)
jax_env = utils.AutoResetWrapper(jax_env)


class KeyParser(NamedTuple):
    int_to_action_str = {
        action.value: action.name for action in jax_env.action_enum()}
    key_to_action = {
        'ArrowUp': 'up',
        'ArrowLeft': 'left',
        'ArrowRight': 'right',
        'ArrowDown': 'down',
        'c': 'continue',
        'd': 'done',
    }

    def valid_key(self, key: str):
        return key in self.key_to_action.keys()

    def action(self, key: str):
        action_name = self.key_to_action.get(key)
        if action_name:
            for action_int, action_str in self.int_to_action_str.items():
                if action_str == action_name:
                    return action_int
        return None


def get_timestep_output(
        stage,
        timestep,
        encode_locally: bool = False,
        **kwargs):
    if encode_locally:
        raise NotImplementedError
        # state_image = stage.render_fn(timestep, env_params)
        # processed_image = encode_image(state_image)
        # return None, processed_image
    else:
        del stage
        state = jax_utils.serialize_pytree(timestep.state, jsonify=False)

        # Keys to keep
        keys_to_keep = ['grid', 'agent_pos', 'agent_dir']

        # Filtered dictionary
        state = {key: state[key] for key in keys_to_keep if key in state}

        return state, None

 
def task_name(timestep):
    category = keys[timestep.state.task_object]
    return f"<b>GOAL: {category}</b>"


def evaluate_success(timestep):
    return int(timestep.reward > .8)

keyparser = KeyParser()

web_env = jax_utils.JaxWebEnvironment(
    env=jax_env,
    keyparser=keyparser,
    timestep_output_fn=get_timestep_output,
    evaluate_success_fn=evaluate_success,
    task_name_fn=task_name,
)

############
# Stages
############


SHORT_PREP = 10
SHORT_ACTION = 10
LONG_PREP = 120
LONG_ACTION = 60
default_env_caption_done = """
<span style="font-weight: bold; font-size: 1.25em;">Movement</span>:<br>
up, down, left, right arrows.
<br><br>
You can press 'd' to finish an episode.
"""

default_env_caption = """
<span style="font-weight: bold; font-size: 1.25em;">Movement</span>:<br>
up, down, left, right arrows.
"""


def make_eval_prep(env_params, title, seconds):
    return webrl.EnvironmentStage(
        'env.html',
        title=title,
        subtitle="""
        Get ready.
        <br><br>
        """,
        env_params=env_params.replace(
            p_test_sample_train=0.,
            terminate_with_done=1,
        ),
        #render_fn=render_timestep_no_obj,
        min_success=1,
        max_episodes=1,
        envcaption=default_env_caption_done,
        seconds=seconds if DEBUG_APP else 5,
        show_progress=False,
        show_goal=True if DEBUG_APP else False,
    )


def make_eval_action(env_params, title, seconds):
    return webrl.EnvironmentStage(
        'env.html',
        title=title,
        subtitle="Use 'd' to indicate when you are done with the stage.",
        env_params=env_params.replace(
            p_test_sample_train=0.,
            terminate_with_done=1,
        ),
        #render_fn=render_timestep_no_obj,
        min_success=1,
        max_episodes=1,
        envcaption=default_env_caption_done,
        seconds=seconds,
        show_progress=False,
        restart=False,
    )


def make_block(
        get_ready_time: int,
        eval_time: int,
        i: int,
        n: int,
        env_params,
        min_success: int = 20,
        max_episodes: int = 200):

    def _make_sublock():
        block = [
            webrl.Stage(
                'explanation.html',
                title=f"Section {i}/{n}",
                body="Please learn to perform these training tasks."
            ),
            webrl.EnvironmentStage(
                'env.html',
                title="Training",
                env_params=env_params.replace(p_test_sample_train=1.),
                #render_fn=render_timestep,
                min_success=1 if DEBUG_APP else min_success,
                max_episodes=1 if DEBUG_APP else max_episodes,
                envcaption=default_env_caption
            ),
            make_eval_prep(
                env_params=env_params, title='Preparation', seconds=get_ready_time),
            make_eval_action(
                env_params=env_params, title='Action', seconds=eval_time),
        ]
        return block
    block = _make_sublock()
    return block

stages = [
    ############################
    # Practice
    ############################
    webrl.Stage(
        'explanation.html',
        title="Practice 1",
        body="""
        You will practice learning how to interact with the environment.
        <br><br>
        You can control the red triangle with the arrow keys on your keyboard.
        <br><br>
        Your goal is to move it to the goal object.
        """
    ),
    webrl.EnvironmentStage(
        'env.html',
        title="Practice 1",
        subtitle="""
        You can control the red triangle with the arrow keys on your keyboard.
        <br><br>
        Your goal is to move it to the goal object.
        """,
        env_params=practice_env_params.replace(p_test_sample_train=1.),
        #render_fn=render_timestep,
        min_success=1 if DEBUG_APP else 5,
        max_episodes=1 if DEBUG_APP else 5,
        envcaption=default_env_caption
    ),
    webrl.Stage(
        'explanation.html',
        title='Practice 2',
        body="""
            Now, you'll practice the evaluation phase.
            <br> There are 2 stages.
            In both stages you will see the map, but objects will be invisible.
            <br><br>
            <br> Stage 1: preparation. You can do whatever you want to prepare.
            <br> Stage 2: action. You must obtain the goal object.
            <br><br>
            Both stages will have a timer that will add either a small or large amount of time pressure.
            <br> Use 'd' to indicate when you are done with the stage.
            """,
    ),
    make_eval_prep(
        env_params=practice_env_params, title='Eval preparation practice', seconds=2),
    make_eval_action(
        env_params=practice_env_params, title='Eval action practice', seconds=5),
    ############################
    # Block 1:
    # 20 trials
    ############################
    *make_block(
        get_ready_time=SHORT_PREP,
        eval_time=SHORT_ACTION,
        i=1, n=4,
        env_params=env1_params,
        min_success=20,
        max_episodes=200,
    ),
    ############################
    # Block 2
    # 20 trials
    ############################
    *make_block(
        get_ready_time=SHORT_PREP,
        eval_time=LONG_ACTION,
        i=2, n=4,
        env_params=env2_params,
        min_success=20,
        max_episodes=200,
    ),
    ############################
    # Block 3
    # 20 trials
    ############################
    *make_block(
        get_ready_time=LONG_PREP,
        eval_time=SHORT_ACTION,
        i=3, n=4,
        env_params=env3_params,
        min_success=20,
        max_episodes=200,
    ),
    *make_block(
        get_ready_time=LONG_PREP,
        eval_time=LONG_ACTION,
        i=4, n=4,
        env_params=env4_params,
        min_success=20,
        max_episodes=200,
    ),
    ############################
    # Done
    ############################
    webrl.Stage('done.html',
                    title='Experiment Finished',
                    subtitle='Please wait as data is uploaded.'
                    ),
]
