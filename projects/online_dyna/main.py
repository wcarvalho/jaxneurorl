from typing import NamedTuple

import argparse
from base64 import b64encode
from datetime import timedelta
from dotenv import load_dotenv
from flask import Flask, render_template, session
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
import utils
import keyroom

load_dotenv()
stage_list = []
interaction_list = []

# Parse command line arguments
parser = argparse.ArgumentParser(description='Your app description')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
parser.add_argument('--debug_seed', type=int, default=1,
                    help='Seed for debugging')
args = parser.parse_args()

DEBUG = args.debug
DEBUG_SEED = args.debug_seed

############
# Set up environment
############
with open("maze_pairs.json", "r") as file:
    maze_config = json.load(file)[0]

env = utils.KeyRoomUpDownLeftRight()

# Action mappings
default_env_params = env.default_params(
    maze_config=keyroom.shorten_maze_config(maze_config, 3))
dummy_rng = jax.random.PRNGKey(0)
dummy_action = 0
default_timestep = env.reset(dummy_rng, default_env_params)
env.step(
    dummy_rng, default_timestep, dummy_action, default_env_params)



class KeyParser(NamedTuple):
    int_to_action_str = {
        0: 'up',
        1: 'right',
        2: 'down',
        3: 'left',
        4: 'pickup',
        5: 'put_down',
        6: 'toggle'
    }
    key_to_action = {
        'w': 'up',
        'a': 'left',
        'd': 'right',
        's': 'down',
        'p': 'pickup',
        'l': 'put_down',
        'o': 'toggle',
        'c': 'continue',
    }

    def valid_key(self, key: str):
        return key in self.key_to_action.keys()

    def action(self, key: str):
        action_name = self.key_to_action.get(key)
        if action_name:
            action_name_to_int = {v: k for k,
                                  v in self.int_to_action_str.items()}
            int_action = action_name_to_int[action_name]
            return int_action
        return None

keyparser = KeyParser()


############
# Set up stages
############
default_env_caption = """
        Press 'W', 'A', 'S', 'D', 'P', 'L', 'O' to interact with the environment.
        <br><br>
        W=up, A=left, D=right, S=down.<br>
        P=pick up, L=put down, O=open.
        """

get_readies = [3, 10]  # small, large
eval_times = [3, 10]  # small, large
def make_block(
        get_ready_time: int,
        eval_time: int,
        min_success: int = 20,
        max_episodes: int = 200):

    def _make_sublock():
        block = [
            utils.Stage(
                'explanation.html',
                title="Training",
                body="Please learn to perform these training tasks."
            ),
            utils.Stage(
                'env.html',
                title="Training",
                type='interaction',
                #subtitle="goal object in a different room",
                env_params=default_env_params.replace(
                    train_multi_probs=1.),
                render_fn=utils.render_map,
                min_success=1 if DEBUG else min_success,
                max_episodes=3 if DEBUG else max_episodes,
                envcaption=default_env_caption
                ),
            utils.Stage(
                'explanation-timed.html',
                title='Evaluation',
                subtitle="Pick the key which will get the object.",
                body="""Get ready.""",
                seconds=get_ready_time,
                ),
            utils.Stage(
                'env.html',
                title='Evaluation',
                subtitle="Pick the key which will get the object.",
                type='1shot',
                env_params=default_env_params.replace(
                    train_multi_probs=1.,
                    training=False,
                    time_limit=1,
                ),
                render_fn=utils.render_keys,
                show_progress=False,
                seconds=eval_time,
                ),
        ]
        return block
    block = _make_sublock()
    return block



stages = [
    utils.Stage('consent.html'),
    ############################
    # Practice
    ############################
    utils.Stage(
        'explanation.html',
        title="Practice 1 - same room",
        body="""
        In this section of the experiment, you'll practice to understand how the environment works. First, you'll practice getting the object in the same room.
        <br><br>
        Please click the right arrow when you are ready.
        """
        ),
    utils.Stage(
        'env.html',
        title="Practice 1 - same room",
        subtitle="goal object in the same room",
        type='interaction',
        env_params=default_env_params.replace(train_multi_probs=0.),
        render_fn=utils.render_map,
        min_success=1 if DEBUG else 10,
        max_episodes=3 if DEBUG else 50,
        envcaption=default_env_caption,
        ),
    utils.Stage(
        'explanation.html',
        title="Practice 2 - multiroom",
        body="""
        Now, you'll practice getting an object when it's in another room.
        <br><br>
        Please click the right arrow when you are ready.
        """
        ),
    utils.Stage(
        'env.html',
        title="Practice 2 - multiroom",
        subtitle="goal object in a different room",
        type='interaction',
        env_params=default_env_params.replace(train_multi_probs=1.),
        render_fn=utils.render_map,
        min_success=1 if DEBUG else 10,
        max_episodes=3 if DEBUG else 50,
        envcaption=default_env_caption
        ),
    utils.Stage(
        'explanation.html',
        title='Practice 3: 1-shot',
        body="""
            Now, you'll practice doing a 1-shot query.
            <br><br>
            Please click the right arrow when you are ready.
            """,
        ),
    utils.Stage(
        'explanation-timed.html',
        title='Practice 3: 1-shot',
        body="""Get ready.""",
        seconds=5,
        ),
    utils.Stage(
        'env.html',
        title='Practice 3: 1-shot',
        subtitle="Pick the key which will get this object.",
        type='1shot',
        env_params=default_env_params.replace(
            train_multi_probs=1.,
            training=False,
            time_limit=1,
            ),
        render_fn=utils.render_keys,
        show_progress=False,
        seconds=10,
        ),
    ############################
    # Block 1:
    # 20 trials
    ############################
    *make_block(
        get_ready_time=3,
        eval_time=3,
        min_success=20,
        max_episodes=200,
    ),
    ############################
    # Block 2
    # 20 trials
    ############################
    *make_block(
        get_ready_time=3,
        eval_time=10,
        min_success=20,
        max_episodes=200,
    ),
    ############################
    # Block 3
    # 20 trials
    ############################
    *make_block(
        get_ready_time=10,
        eval_time=3,
        min_success=20,
        max_episodes=200,
    ),
    *make_block(
        get_ready_time=10,
        eval_time=10,
        min_success=20,
        max_episodes=200,
    ),
    ############################
    # Done
    ############################
    utils.Stage('done.html',
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

def render(timestep):
    return utils.render(
        np.asarray(timestep.state.grid),
        timestep.state.agent,
        0,
        tile_size=20)


def encode_image(state_image):
    buffer = io.BytesIO()
    Image.fromarray(state_image.astype('uint8')).save(buffer, format="JPEG")
    encoded_image = b64encode(buffer.getvalue()).decode('ascii')
    return 'data:image/jpeg;base64,' + encoded_image


def get_task_name(timestep):
    room_idx = int(timestep.state.goal_room_idx)
    object_idx = int(timestep.state.task_object_idx)

    category, color = maze_config['pairs'][room_idx][object_idx]
    return f"<b>GOAL</b>: pickup the {color} {category}"

def evaluate_success(timestep):
    return int(timestep.reward > .8)

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

    stage = stages[session['stage_idx']]
    state_image = stage.render_fn(timestep, env_params, rng_)

    session['timestep'] = timestep
    return state_image

def serialize(pytree):
    pytree = serialization.to_state_dict(pytree)
    pytree = utils.array_to_python(pytree)
    return json.dumps(pytree)


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
    new_row = {
        "stage_idx": int(stage_idx),
        "image_seen_time": str(socket_json['imageSeenTime']),
        "key_press_time": str(socket_json['keydownTime']),
        "key": str(socket_json['key']),
        "action": int(keyparser.action(socket_json['key'])),
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

    emit('update_html_fields', {
        'title': stages[session['stage_idx']].title,
        'subtitle': stages[session['stage_idx']].subtitle,
        'body': f"Uploaded {ninteraction} interactions from {nstage} stages. Thank you. You may now close the browser",
    })


def update_html_fields(**kwargs):
    emit('update_html_fields', {
        'title': stages[session['stage_idx']].title,
        'subtitle': stages[session['stage_idx']].subtitle,
        'body': stages[session['stage_idx']].body,
        'envcaption': stages[session['stage_idx']].envcaption,
        **kwargs,
    })

    seconds = stages[session['stage_idx']].seconds
    if seconds:
        emit('start_timer', {'seconds': seconds})


def update_env_html_fields(**kwargs):

    stage_idx = session['stage_idx']
    stage = stages[stage_idx]
    stage_info = session['stage_infos'][stage_idx]

    subtitle = stage.subtitle
    if stage.show_progress:
        subtitle += f"<br>Successes: {stage_info.num_success}/{stage.min_success}"
        subtitle += f"<br>Episodes: {stage_info.ep_idx}/{stage.max_episodes}"
    emit('update_html_fields', {
        'title': stage.title,
        'subtitle': subtitle,
        'taskDesc': get_task_name(session['timestep']),
        'body': stage.body,
        'envcaption': stage.envcaption,
        **kwargs,
    })


def start_env_interaction_stage():
    """New stage begins."""
    stage = stages[session['stage_idx']]
    template_file = stages[session['stage_idx']].html
    env_params = stages[session['stage_idx']].env_params

    print('about to reset env')
    reset_environment(env_params)

    state_image = utils.render_map(session['timestep'])
    encoded_image = encode_image(state_image)

    emit('update_content', {
        'content': render_template(template_file),
    })
    print('loading new index')
    emit('action_taken', {
        'image': encoded_image,
    })
    print('loading image')
    update_env_html_fields()
    print('adding html content')


def handle_interaction_phase(json):

    key = json['key']
    if not keyparser.valid_key(key):
        return

    stage_idx = session['stage_idx']
    stage_info = session['stage_infos'][stage_idx]

    env_params = stages[stage_idx].env_params
    if not session['timestep'].last():
        # update database with image, action, + times of each
        add_interaction_to_db(json, stage_idx,
                              session['timestep'], session['rng'], session['user_seed'])
        print('add_interaction_to_db')

        # take action
        state_image = take_action(key, env_params)
        encoded_image = encode_image(state_image)

        emit('action_taken', {
            'image': encoded_image,
        })
        print('next state')

        # is the NEXT time-step going to be last?
        if session['timestep'].last():
            # evaluate success and update stage/info desc accordingly
            success = evaluate_success(session['timestep'])
            session['stage_infos'][stage_idx] = stage_info.replace(
                ep_idx=stage_info.ep_idx+1,
                num_success=stage_info.num_success + success,
                t=stage_info.t+1,
            )
            label = 'SUCCESS' if success else 'FAILED'
            color = 'green' if success else 'red'
            label = f'<span style = "color: {color};">{label}</span >'
            update_env_html_fields(
                taskDesc=f"{label}! restarting. press 'c' to continue.",
            )
            print('updated stage_info at end')
        else:
            session['stage_infos'][stage_idx] = stage_info.replace(
                t=stage_info.t+1,
            )
            print('updated stage_info.t')

    else:
        # if final time-step, need to press 'c' to continue.
        if key != 'c':
            return

        ###################
        # check if this stage is over
        ###################
        ep_idx = stage_info.ep_idx
        num_success = stage_info.num_success

        stage = stages[stage_idx]
        max_episodes = stage.max_episodes
        min_success = stage.min_success

        achieved_min_success = num_success >= min_success
        achieved_max_episodes = ep_idx > max_episodes

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
            state_image = utils.render_map(session['timestep'])
            encoded_image = encode_image(state_image)
            update_env_html_fields(
                taskDesc=get_task_name(session['timestep']),
            )
            emit('action_taken', {
                'image': encoded_image,
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
    state_image = utils.objects_with_number(keys)
    encoded_image = encode_image(state_image)

    emit('update_content', {
        'content': render_template(template_file),
    })
    emit('action_taken', {
        'image': encoded_image,
    })

    seconds = stages[session['stage_idx']].seconds
    if seconds:
        emit('start_timer', {'seconds': seconds})
    kwargs = {}
    if DEBUG:
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

def start_env_stage():
    stage = stages[session['stage_idx']]
    if stage.type == 'interaction':
        start_env_interaction_stage()
    elif stage.type == '1shot':
        start_env_1shot_phase()

def end_1shot_timer():
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


def advance_to_next_stage():
    # update stage idx
    session['stage_idx'] += 1
    session['stage_idx'] = min(session['stage_idx'], len(stages)-1)

    # next template file
    template_file = stages[session['stage_idx']].html

    # Emit the 'stage_advanced' event to the client
    emit('stop_timer')

    # update content
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

CORS(app)  # Enable CORS for all routes

app.secret_key = 'some secret'
# Adjust the time as needed
app.permanent_session_lifetime = timedelta(days=30)

app.json.default = utils.encode_json
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
    if DEBUG:
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
    session['stage_idx'] = 1
    return render_template(
        'index.html', template_file=stages[session['stage_idx']].html)


@socketio.on('request_update')
def handle_request_update():
    """Always called 3rd immediately after `start_experiment`. Need separate function to listen for rendering a new template. We'll now update the html content."""

    # NOTE: we want to store this in session because will be changing
    stage_infos = [
        utils.StageInfo(stage) if 'env' in stage.html else None for stage in stages]
    session['stage_infos'] = stage_infos

    # Check if the stage index is set and then emit the update_html_fields event.
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
    if direction == 'left':
        session['stage_idx'] -= 1
        session['stage_idx'] = max(1, session['stage_idx'])
    elif direction == 'right':
        session['stage_idx'] += 1
        session['stage_idx'] = min(session['stage_idx'], len(stages)-1)
    else:
        raise NotImplementedError

    template_file = stages[session['stage_idx']].html
    if 'env' in template_file:
        start_env_stage()
    else:
        emit('update_content', {
            'content': render_template(template_file)
        })
        update_html_fields()
        if 'done' in template_file:
            save_interactions_on_session_end()

@socketio.on('key_pressed')
def handle_key_press(json):
    """This happens INSIDE a stage"""
    print('key pressed:', json['key'])

    stage = stages[session['stage_idx']]
    if stage.type == 'interaction':
        handle_interaction_phase(json)
    elif stage.type == '1shot':
        handle_1shot_phase(json)


@socketio.on('timer_finished')
def on_timer_finish():
    stage = stages[session['stage_idx']]
    if stage.type == '1shot':
        end_1shot_timer()
    else:
        advance_to_next_stage()

# Run the Flask app
if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
