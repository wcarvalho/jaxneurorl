import sys
import time
sys.path.append('ml')

from typing import NamedTuple

from base64 import b64encode
from datetime import timedelta
from dotenv import load_dotenv
from flask import Flask, render_template, session
from flask import Flask, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from flax import serialization
from google.cloud import storage
import io
import jax
import jax.numpy as jnp
import json
import random
import numpy as np
import os
from PIL import Image

from functools import partial
import web_utils
from ml.housemaze import levels
from ml.housemaze import renderer
import ml.housemaze_env as maze
from ml import housemaze_utils
import mazes


load_dotenv()
stage_list = []
interaction_list = []

TILE_SIZE = 32
DEBUG_APP = int(os.environ.get('DEBUG_APP', 0))
DEBUG_SEED = os.environ.get('DEBUG_SEED', 1)

############
# Set up environment
############

file = 'ml/housemaze_list_of_groups.npy'
list_of_groups = np.load(file)
group_set = list_of_groups[0]
num_groups = 3
group_set = group_set[:num_groups]

def make_env_params(maze_str: str):
    return mazes.make_env_params(
        maze_str=maze_str,
        group_set=group_set,
        ).replace(
            training=False,
            terminate_with_done=True if DEBUG_APP else False,
            )


practice_env_params = make_env_params(mazes.maze0)
env1_params = make_env_params(mazes.maze1)
env2_params = make_env_params(mazes.maze2)
env3_params = make_env_params(mazes.maze3)
env4_params = make_env_params(mazes.maze4)


image_data = housemaze_utils.load_image_dict(
    'ml/housemaze/image_data.pkl')
json_image_data = web_utils.convert_to_serializable(image_data)

task_objects = group_set.reshape(-1)
task_runner = maze.TaskRunner(
    task_objects=task_objects)
keys = image_data['keys']

env = maze.HouseMaze(
    task_runner=task_runner,
    num_categories=len(keys),
    use_done=True,
)
env = housemaze_utils.AutoResetWrapper(env)
################



# RNG
dummy_rng = jax.random.PRNGKey(0)
dummy_action = 0
default_timestep = env.reset(dummy_rng, practice_env_params)
env.step(
    dummy_rng, default_timestep, dummy_action, practice_env_params)


def get_timestep_output(stage, timestep, env_params, encode_locally: bool = False):
    stage = stages[session['stage_idx']]
    if encode_locally:
        state_image = stage.render_fn(timestep, env_params)
        processed_image = encode_image(state_image)
        return None, processed_image
    else:
        import jax
        state = jax.tree_map(np.asarray, timestep.state)
        state = serialize(state, jsonify=False)
        state = web_utils.convert_to_serializable(state)

        # Keys to keep
        keys_to_keep = ['grid', 'agent_pos', 'agent_dir']

        # Filtered dictionary
        state = {key: state[key] for key in keys_to_keep if key in state}

        return state, None


def render_timestep(timestep, env_params, **kwargs):
    del env_params
    image = renderer.create_image_from_grid(
        timestep.state.grid,
        timestep.state.agent_pos,
        timestep.state.agent_dir,
        image_data,
        **kwargs)
    return np.asarray(image)


render_timestep_no_obj = partial(
    render_timestep, include_objects=False)

def get_task_name(timestep):
    category = keys[timestep.state.task_object]
    return f"<b>GOAL: {category}</b>"


def evaluate_success(timestep):
    return int(timestep.reward > .8)


class KeyParser(NamedTuple):
    int_to_action_str = {
        action.value: action.name for action in env.action_enum()}
    key_to_action = {
        #'w': 'up',
        'ArrowUp': 'up',
        #'a': 'left',
        'ArrowLeft': 'left',
        #'d': 'right',
        'ArrowRight': 'right',
        #'s': 'down',
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

keyparser = KeyParser()


############
# Set up stages
############
default_env_caption = """
<span style="font-weight: bold; font-size: 1.25em;">Movement</span>:<br>
up, down, left, right arrows.
<br><br>
You can press 'd' to finish an episode.
"""

SHORT_PREP = 10
SHORT_ACTION = 10
LONG_PREP = 120
LONG_ACTION = 60

def make_eval_prep(env_params, title, seconds):
    return web_utils.Stage(
        'env.html',
        title=title,
        subtitle="""
        Get ready.
        <br><br>
        Use 'd' to indicate when you are done with the stage.
        <br>(not recommended)
        """,
        type='pause',
        env_params=env_params.replace(
            p_test_sample_train=0.,
            terminate_with_done=True,
            ),
        render_fn=render_timestep_no_obj,
        min_success=1,
        max_episodes=1,
        envcaption=default_env_caption,
        seconds=seconds if DEBUG_APP else 5,
        show_progress=False,
        show_goal=True if DEBUG_APP else False,
    )


def make_eval_action(env_params, title, seconds):
    return web_utils.Stage(
        'env.html',
        title=title,
        subtitle="Use 'd' to indicate when you are done with the stage.",
        type='interaction',
        env_params=env_params.replace(
            p_test_sample_train=0.,
            terminate_with_done=True,
        ),
        render_fn=render_timestep_no_obj,
        min_success=1,
        max_episodes=1,
        envcaption=default_env_caption,
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
            web_utils.Stage(
                'explanation.html',
                title=f"Section {i}/{n}",
                body="Please learn to perform these training tasks."
            ),
            web_utils.Stage(
                'env.html',
                title="Training",
                type='interaction',
                env_params=env_params.replace(p_test_sample_train=1.),
                render_fn=render_timestep,
                min_success=1 if DEBUG_APP else min_success,
                max_episodes=3 if DEBUG_APP else max_episodes,
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
    web_utils.Stage('consent.html'),
    ############################
    # Practice
    ############################
    web_utils.Stage(
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
    web_utils.Stage(
        'env.html',
        title="Practice 1",
        subtitle="""
        You can control the red triangle with the arrow keys on your keyboard.
        <br><br>
        Your goal is to move it to the goal object.
        """,
        type='interaction',
        env_params=practice_env_params.replace(p_test_sample_train=1.),
        render_fn=render_timestep,
        min_success=1 if DEBUG_APP else 5,
        max_episodes=3 if DEBUG_APP else 5,
        envcaption=default_env_caption
        ),
    web_utils.Stage(
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
    web_utils.Stage('done.html',
            title='Experiment Finished',
            subtitle='Please wait as data is uploaded.'
            ),
]

############
# Helper functions
############


def rng_from_jax(rng):
    return tuple(int(i) for i in rng)


def rng_to_jax(rng_tuple):
    return jnp.array(rng_tuple, dtype=jnp.uint32)


def split_rng():
    rng = rng_to_jax(session['rng'])
    rng, rng_ = jax.random.split(rng)
    session['rng'] = rng_from_jax(rng)
    return rng_


def encode_image(state_image):
    buffer = io.BytesIO()
    Image.fromarray(state_image.astype('uint8')).save(buffer, format="JPEG")
    encoded_image = b64encode(buffer.getvalue()).decode('ascii')
    return 'data:image/jpeg;base64,' + encoded_image

def make_title(stage, session, debug):
    stage_idx = session['stage_idx']
    if debug:
        return f"{stage_idx}/{len(stages)}. {stage.title}"
    else:
        return f"{stage.title}"

def reset_environment(env_params):
    rng_ = split_rng()
    timestep = env.reset(rng_, env_params)
    session['timestep'] = timestep


def take_action(action_key, env_params):
    rng_ = split_rng()

    action = keyparser.action(action_key)
    if action is not None:
        timestep = env.step(rng_, session['timestep'], action, env_params)
    else:
        if 'timestep' in session:
            timestep = session['timestep']
        else:
            raise RuntimeError("no previous time-step to re-emit and no action taken?")

    #stage = stages[session['stage_idx']]
    #state_image = stage.render_fn(timestep, env_params)

    session['timestep'] = timestep
    #return state_image

def serialize(pytree, jsonify: bool = True):
    pytree = serialization.to_state_dict(pytree)
    pytree = web_utils.array_to_python(pytree)
    if jsonify:
        return json.dumps(pytree)
    return pytree


def add_stage_to_db(stage_idx, stage_infos, user_seed):
    stage_info = stage_infos[stage_idx]
    stage = stages[stage_idx]
    stage = stage.replace(render_fn=None)
    stage = serialize(stage)
    new_row = {
        "stage_idx": stage_idx,
        'stage': stage,
        't': stage_info.t,
        'ep_idx': stage_info.ep_idx,
        'num_success': stage_info.num_success,
        'unique_id': user_seed,
    }
    stage_list.append(new_row)
    print('added stage row', len(stage_list))


def add_interaction_to_db(socket_json, stage_idx, timestep, rng, user_seed):
    timestep = timestep.replace(observation=None)
    timestep = serialize(timestep)
    action = keyparser.action(socket_json['key'])
    action = int(action) if action else action
    new_row = {
        "stage_idx": int(stage_idx),
        "image_seen_time": str(socket_json['imageSeenTime']),
        "key_press_time": str(socket_json['keydownTime']),
        "key": str(socket_json['key']),
        "action": action,
        "timestep": timestep,
        "rng": list(rng_from_jax(rng)),
        'unique_id': int(user_seed),
    }
    interaction_list.append(new_row)
    print('added interaction row:', len(interaction_list))


def initialize_storage_client():
    storage_client = storage.Client.from_service_account_json(
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
    bucket_name = 'human-web-rl_cloudbuild'
    bucket = storage_client.bucket(bucket_name)
    return bucket


def save_list_to_gcs(interaction_list, filename):
    bucket = initialize_storage_client()
    blob = bucket.blob(filename)
    blob.upload_from_string(data=json.dumps(
        interaction_list), content_type='application/json')
    print(f'Saved {filename} in bucket {bucket.name}')

def save_interactions_on_session_end():
    ninteraction = len(interaction_list)
    nstage = len(stage_list)
    print(f"Uploading {ninteraction} interactions from {nstage} stages.")

    unique_id = session['user_seed']
    filename = f'online_dyna/interactions_{unique_id}.json'
    save_list_to_gcs(interaction_list, filename)

    filename = f'online_dyna/stage_infos_{unique_id}.json'
    save_list_to_gcs(stage_list, filename)


    stage = stages[session['stage_idx']]
    emit('update_html_fields', {
        'title': make_title(stage, session, DEBUG_APP),
        'subtitle': stages[session['stage_idx']].subtitle,
        'body': f"Uploaded {ninteraction} interactions from {nstage} stages. Thank you. You may now close the browser",
    })


def update_html_fields(**kwargs):
    stage_idx = session['stage_idx']
    stage = stages[stage_idx]
    emit('update_html_fields', {
        'title': make_title(stage, session, DEBUG_APP),
        'subtitle': stage.subtitle,
        'body': stage.body,
        'envcaption': stage.envcaption,
        **kwargs,
    })


def update_env_html_fields(**kwargs):

    stage_idx = session['stage_idx']
    stage = stages[stage_idx]
    stage_info = session['stage_infos'][stage_idx]

    subtitle = stage.subtitle
    if stage.show_progress:
        subtitle += f"<br>Successes: {stage_info.num_success}/{stage.min_success}"
        subtitle += f"<br>Episodes: {stage_info.ep_idx}/{stage.max_episodes}"
    
    task = get_task_name(session['timestep']) if stage.show_goal else ''
    emit('update_html_fields', {
        'title': make_title(stage, session, DEBUG_APP),
        'subtitle': subtitle,
        'taskDesc': task,
        'body': stage.body,
        'envcaption': stage.envcaption,
        **kwargs,
    })
    #seconds = stages[session['stage_idx']].seconds
    #if seconds:
    #    print('starting timer: update_env_html_fields')
    #    emit('start_timer', {'seconds': seconds})


def start_env_interaction_stage():
    """New stage begins."""
    stage = stages[session['stage_idx']]
    template_file = stages[session['stage_idx']].html
    env_params = stages[session['stage_idx']].env_params

    if stage.restart:
        print('about to reset env')
        reset_environment(env_params)

    #state_image = stage.render_fn(session['timestep'], env_params)
    #encoded_image = encode_image(state_image)
    raw_state, encoded_image = get_timestep_output(
        stage=stages[session['stage_idx']],
        timestep=session['timestep'],
        env_params=stages[session['stage_idx']].env_params,
        encode_locally=False,
    )

    emit('update_content', {
        'content': render_template(template_file),
    })

    print('loading new index')
    emit('action_taken', {
        'image': encoded_image,
        'state': raw_state,
    })
    print('loading image')
    update_env_html_fields()
    print('adding html content')

def maybe_start_count_down():
    stage = stages[session['stage_idx']]
    seconds = stage.seconds
    if seconds:
        count_down_started = stage.count_down_started
        if not count_down_started:
            print('starting timer: start_env_stage')
            emit('start_timer', {'seconds': seconds})

        stages[session['stage_idx']] = stage.replace(
            count_down_started=True
        )

def handle_interaction_phase(json):

    key = json['key']

    stage_idx = session['stage_idx']
    stage_info = session['stage_infos'][stage_idx]

    env_params = stages[stage_idx].env_params
    if not session['timestep'].last():
        if not keyparser.valid_key(key):
            return

        # update database with image, action, + times of each
        add_interaction_to_db(json, stage_idx,
                              session['timestep'], session['rng'], session['user_seed'])
        print('add_interaction_to_db')

        # take action
        take_action(key, env_params)
        raw_state, encoded_image = get_timestep_output(
            stage=stages[session['stage_idx']],
            timestep=session['timestep'],
            env_params=stages[session['stage_idx']].env_params,
            encode_locally=False,
        )
        #encoded_image = encode_image(state_image)

        emit('action_taken', {
            'image': encoded_image,
            'state': raw_state,
        })
        print('last?', session['timestep'].last())
        print('reward?', session['timestep'].reward)

        # is the NEXT time-step going to be last?
        if session['timestep'].last():
            # evaluate success and update stage/info desc accordingly
            success = evaluate_success(session['timestep'])
            session['stage_infos'][stage_idx] = stage_info.replace(
                ep_idx=stage_info.ep_idx+1,
                num_success=stage_info.num_success + success,
                t=stage_info.t+1,
            )
            label = ''
            if stages[session['stage_idx']].show_progress:
                label = 'SUCCESS' if success else 'FAILED'
                color = 'green' if success else 'red'
                label = f'<span style="color: {color}; font-weight: bold; font-size: 1.5em;">{label}!</span><br>'
            update_env_html_fields(
                taskDesc=f"{label}restarting. press any key to continue.",
            )
            print('updated stage_info at end')
        else:
            session['stage_infos'][stage_idx] = stage_info.replace(
                t=stage_info.t+1,
            )
            print('updated stage_info.t')

    else:

        ###################
        # check if this stage is over
        ###################
        ep_idx = stage_info.ep_idx
        num_success = stage_info.num_success

        stage = stages[stage_idx]
        max_episodes = stage.max_episodes
        min_success = stage.min_success

        achieved_min_success = num_success >= min_success
        achieved_max_episodes = ep_idx >= max_episodes

        go_to_next_stage = achieved_min_success or achieved_max_episodes
        # ------------
        # update to next stage
        # ------------
        if go_to_next_stage:
            add_stage_to_db(session['stage_idx'],
                            session['stage_infos'], session['user_seed'])
            advance_to_next_stage()
            print('advanced to next stage')

        # ------------
        # reset environment
        # ------------
        else:
            reset_environment(env_params)
            raw_state, encoded_image = get_timestep_output(
                stage=stages[session['stage_idx']],
                timestep=session['timestep'],
                env_params=stages[session['stage_idx']].env_params,
                encode_locally=False,
            )
            #stage = stages[session['stage_idx']]
            #state_image = stage.render_fn(session['timestep'], stage.env_params)
            #encoded_image = encode_image(state_image)
            update_env_html_fields(
                taskDesc=get_task_name(session['timestep']),
            )
            emit('action_taken', {
                'image': encoded_image,
                'state': raw_state,
            })
            print('reset env')

def start_env_1shot_phase():

    template_file = stages[session['stage_idx']].html
    env_params = stages[session['stage_idx']].env_params
    keys = env_params.maze_config['keys']

    reset_environment(env_params)
    rng_ = split_rng()
    permutation = jax.random.permutation(rng_, keys.shape[0])
    session['permutation'] = permutation

    keys = keys[permutation]
    session['choices'] = keys
    state_image = web_utils.objects_with_number(keys)
    encoded_image = encode_image(state_image)

    emit('update_content', {
        'content': render_template(template_file),
    })
    emit('action_taken', {
        'image': encoded_image,
        'state': raw_state,
    })

    kwargs = {}
    if DEBUG_APP:
        goal_room_idx = int(session['timestep'].state.goal_room_idx)
        kwargs['envcaption'] = f'correct key: {permutation[goal_room_idx]}'

    update_env_html_fields(**kwargs)

def handle_1shot_phase(json):
    key = json['key']
    ###################
    # evaluate whether successful
    ###################

    if not key.isnumeric():
        print(f"{key} is not numeric")
        return
    choice_idx = int(key) - 1
    choices = session['choices']
    if not choice_idx < len(choices): 
        print(f'{choice_idx}>{len(choices)}')
        return

    # what is the goal key? aliased with goal room.
    goal_room_idx = int(session['timestep'].state.goal_room_idx)
    chosen_room = int(session['permutation'][choice_idx])
    success = int(goal_room_idx == chosen_room)
    print("Success:", success)
    ###################
    # store data and move to next stage
    ###################
    stage_idx = session['stage_idx']
    rng = session['rng']
    #-------------
    # interactions
    #-------------
    user_seed = session['user_seed']
    timestep = session['timestep'].replace(observation=None)
    timestep = serialize(timestep)
    new_row = {
        "stage_idx": int(stage_idx),
        "image_seen_time": str(json['imageSeenTime']),
        "key_press_time": str(json['keydownTime']),
        "key": str(json['key']),
        "action": int(json['key']),
        "timestep": timestep,
        "rng": list(rng_from_jax(rng)),
        'unique_id': int(user_seed),
    }
    interaction_list.append(new_row)

    #---------------
    # stages
    #---------------
    stage = stages[stage_idx]
    stage = stage.replace(render_fn=None)
    stage = serialize(stage)
    new_row = {
        "stage_idx": stage_idx,
        'stage': stage,
        't': 1,
        'ep_idx': 1,
        'num_success': success,
        'unique_id': int(user_seed),
    }
    stage_list.append(new_row)
    advance_to_next_stage()


def handle_pause_phase(json):
    key = json['key']
    ###################
    # evaluate whether successful
    ###################

    if not key.isnumeric():
        print(f"{key} is not numeric")
        return
    choice_idx = int(key) - 1
    choices = session['choices']
    if not choice_idx < len(choices):
        print(f'{choice_idx}>{len(choices)}')
        return

    # what is the goal key? aliased with goal room.
    goal_room_idx = int(session['timestep'].state.goal_room_idx)
    chosen_room = int(session['permutation'][choice_idx])
    success = int(goal_room_idx == chosen_room)
    print("Success:", success)
    ###################
    # store data and move to next stage
    ###################
    stage_idx = session['stage_idx']
    rng = session['rng']
    # -------------
    # interactions
    # -------------
    user_seed = session['user_seed']
    timestep = session['timestep'].replace(observation=None)
    timestep = serialize(timestep)
    new_row = {
        "stage_idx": int(stage_idx),
        "image_seen_time": str(json['imageSeenTime']),
        "key_press_time": str(json['keydownTime']),
        "key": str(json['key']),
        "action": int(json['key']),
        "timestep": timestep,
        "rng": list(rng_from_jax(rng)),
        'unique_id': int(user_seed),
    }
    interaction_list.append(new_row)

    # ---------------
    # stages
    # ---------------
    stage = stages[stage_idx]
    stage = stage.replace(render_fn=None)
    stage = serialize(stage)
    new_row = {
        "stage_idx": stage_idx,
        'stage': stage,
        't': 1,
        'ep_idx': 1,
        'num_success': success,
        'unique_id': int(user_seed),
    }
    stage_list.append(new_row)
    advance_to_next_stage()

def start_env_stage():
    stage = stages[session['stage_idx']]
    if stage.type == 'interaction':
        start_env_interaction_stage()
    elif stage.type == '1shot':
        start_env_1shot_phase()
    elif stage.type == 'pause':
        start_env_interaction_stage()
    maybe_start_count_down()

def end_timer_no_interaction():
    ###################
    # store data and move to next stage
    ###################
    stage_idx = session['stage_idx']
    rng = session['rng']
    # -------------
    # interactions
    # -------------
    user_seed = session['user_seed']
    timestep = session['timestep'].replace(observation=None)
    timestep = serialize(timestep)
    new_row = {
        "stage_idx": int(stage_idx),
        "image_seen_time": 'none',
        "key_press_time": 'none',
        "key": 'none',
        "action": -1000,
        "timestep": timestep,
        "rng": list(rng_from_jax(rng)),
        'unique_id': int(user_seed),
    }
    interaction_list.append(new_row)

    # ---------------
    # stages
    # ---------------
    stage = stages[stage_idx]
    stage = stage.replace(render_fn=None)
    stage = serialize(stage)
    new_row = {
        "stage_idx": stage_idx,
        'stage': stage,
        't': 1,
        'ep_idx': 1,
        'num_success': 0,
        'unique_id': int(user_seed),
    }
    stage_list.append(new_row)
    # Define the logic to be executed when the timer finishes
    advance_to_next_stage()


def shift_stage(direction: str):
    if direction == 'left':
        session['stage_idx'] -= 1
        session['stage_idx'] = max(1, session['stage_idx'])
    elif direction == 'right':
        session['stage_idx'] += 1
        session['stage_idx'] = min(session['stage_idx'], len(stages)-1)
    else:
        raise NotImplementedError

    template_file = stages[session['stage_idx']].html
    print("="*50)
    print("STAGE:", stages[session['stage_idx']].title)
    if 'env' in template_file:
        start_env_stage()
    else:
        emit('update_content', {
            'content': render_template(template_file)
        })
        update_html_fields()
        if 'done' in template_file:
            save_interactions_on_session_end()
    maybe_start_count_down()

def advance_to_next_stage():
    # update stage idx
    session['stage_idx'] += 1
    session['stage_idx'] = min(session['stage_idx'], len(stages)-1)

    # next template file
    template_file = stages[session['stage_idx']].html

    # update content
    print("="*50)
    print("STAGE:", stages[session['stage_idx']].title)
    if 'env' in template_file:
        start_env_stage()
    else:
        emit('update_content', {
            'content': render_template(template_file),
        })
        update_html_fields()
        if 'done' in template_file:
            save_interactions_on_session_end()


############
# App
############
app = Flask(__name__)

#CORS(app)  # Enable CORS for all routes

app.secret_key = 'some secret'
# Adjust the time as needed
app.permanent_session_lifetime = timedelta(days=30)

app.json.default = web_utils.encode_json
socketio = SocketIO(app)

# Route for the index page
@app.route('/')
def index():
    """Always Called 1st"""
    session.permanent = True
    if 'user_seed' not in session:
        pass
        # Generate a unique user ID if not present
        # unique_id = uuid.uuid4()
    unique_id = random.getrandbits(32)
    user_seed = int(unique_id)

    # reset the environment early + run reset/step to jit computations
    if DEBUG_APP:
        rng = jax.random.PRNGKey(DEBUG_SEED)
    else:
        rng = jax.random.PRNGKey(user_seed)

    session['unique_id'] = unique_id
    session['user_seed'] = user_seed
    rng = rng_from_jax(rng)
    session['rng'] = rng
    session['stage_idx'] = 0

    return render_template('index.html', template_file='consent.html')


@app.route('/experiment', methods=['POST'])
def start_experiment():
    """Always called 2nd after checkbox is checked to start experiment."""
    if session['stage_idx'] == 0:
        session['stage_idx'] = 1
        return render_template(
            'index.html', template_file=stages[1].html)
    elif session['stage_idx'] == 1:
        return render_template(
            'index.html', template_file=stages[1].html)
    elif session['stage_idx'] > 1:
        stage_list.clear()
        interaction_list.clear()
        print('cleared interaction')
        return render_template('index.html', template_file=stages[1].html)


@socketio.on('request_update')
def handle_request_update():
    """Always called 3rd immediately after `start_experiment`. Need separate function to listen for rendering a new template. We'll now update the html content."""
    emit('load_data', {'image_data': json_image_data})
    # NOTE: we want to store this in session because will be changing
    stage_infos = [
        web_utils.StageInfo(stage) if 'env' in stage.html else None for stage in stages]
    session['stage_infos'] = stage_infos

    # Check if the stage index is set and then emit the update_html_fields event.
    print("="*50)
    print("STAGE:", stages[session['stage_idx']].title)
    if 'stage_idx' in session:
        update_html_fields()
        template_file = stages[session['stage_idx']].html
        if 'env' in template_file:
            start_env_stage()

@socketio.on('record_click')
def handle_record_click(json):
    """Once the experiment has started, the user can go back and forth by clicking arrows.

    This allows for that.
    """
    direction = json['direction']
    print("direction:", direction)
    shift_stage(direction)

@socketio.on('key_pressed')
def handle_key_press(json):
    """This happens INSIDE a stage"""
    print('-'*10)
    print('key pressed:', json['key'])

    stage = stages[session['stage_idx']]
    if stage.type == 'interaction':
        handle_interaction_phase(json)
    elif stage.type == '1shot':
        handle_1shot_phase(json)
    elif stage.type == 'pause':
        if json['key'] == 'd':
            advance_to_next_stage()
        pass  # do no nothing. just ignore keys
    elif stage.type == 'default':
        key = json['key']
        if key in ('ArrowLeft', 'ArrowRight'):
            direction = {
                'ArrowLeft': 'left',
                'ArrowRight': 'right',
            }[key]
            shift_stage(direction)
        else:
            return


@socketio.on('timer_finished')
def on_timer_finish():
    stage = stages[session['stage_idx']]
    print(f'{stage.title}: timer finished')
    emit('stop_timer')
    if stage.type in ('1shot', 'pause'):
        print("end_timer_no_interaction")
        end_timer_no_interaction()
    else:
        print("advance_to_next_stage")
        advance_to_next_stage()

# Run the Flask app
if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
