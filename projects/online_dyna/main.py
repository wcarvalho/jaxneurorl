import gevent
from gevent import monkey
monkey.patch_all()

from typing import NamedTuple

from base64 import b64encode
from datetime import timedelta
from dotenv import load_dotenv
from flask import Flask, render_template, request, session
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from flax import serialization
from google.cloud import firestore
import google.api_core.exceptions
from google.api_core import retry
import io
import jax
import jax.numpy as jnp
import json
import random
import numpy as np
import threading
import os
from PIL import Image
import utils
import keyroom

load_dotenv()
# Create a client
db = firestore.Client()

# Reference an existing document or create a new one in 'your-collection'
stage_info_db = db.collection('online-dyna-stage-info')
interactions_db = db.collection('online-dyna-interactions')

############
# Set up environment
############
with open("maze_pairs.json", "r") as file:
    maze_config = json.load(file)[0]

env = keyroom.KeyRoom()

# Action mappings
default_env_params = env.default_params(
    maze_config=keyroom.shorten_maze_config(maze_config, 3))
dummy_rng = jax.random.PRNGKey(0)
default_timestep = env.reset(dummy_rng, default_env_params)
dummy_action = 0
env.step(
    dummy_rng, default_timestep, dummy_action, default_env_params)



class KeyParser(NamedTuple):
    int_to_action_str = {
        0: 'forward',
        1: 'right',
        2: 'left',
        3: 'pickup',
        4: 'put_down',
        5: 'toggle'
    }
    key_to_action = {
        'w': 'forward',
        'a': 'left',
        'd': 'right',
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
stages = [
    utils.Stage('consent.html'),
    utils.Stage('explanation.html',
          title='Practice',
          body="""
            In this section of the experiment, you'll practice to understand how the environment works. First, you'll practice getting the object in the same room.
            <br><br>
            You will do 5 practice rounds.
            <br><br>
            Please click the right arrow when you are done.
            """
          ),
    utils.Stage('env.html',
          title="Practice 1",
          subtitle="goal object in the same room",
          env_params=default_env_params.replace(train_multi_probs=0.,),
          min_success=1,
          envcaption=default_env_caption,
          ),
    utils.Stage('explanation.html',
          title='Practice 2',
          body="""
            Now, you'll practice getting an object when it's in another room.
            <br><br>
            You will do 5 practice rounds.
            <br><br>
            Please click the right arrow when you are done.
            """
          ),
    utils.Stage('env.html',
          title="Practice 2",
          subtitle="goal object in a different room",
          env_params=default_env_params.replace(train_multi_probs=1.,),
          min_success=1,
          envcaption=default_env_caption
          ),
    utils.Stage('done.html'),
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
    state_image = render(timestep)

    # udpate timestep in session
    session['timestep'] = timestep
    return state_image


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
    state_image = render(timestep)

    session['timestep'] = timestep
    return state_image

def start_env_interaction_stage():
    """New stage begins."""
    template_file = stages[session['stage_idx']].html

    assert 'env' in template_file

    env_params = stages[session['stage_idx']].env_params
    state_image = reset_environment(env_params)
    encoded_image = encode_image(state_image)

    emit('update_content', {
        'content': render_template(template_file),
    })
    emit('action_taken', {
            'image': encoded_image,
        })
    emit('update_html_fields', {
            'title': stages[session['stage_idx']].title,
            'subtitle': stages[session['stage_idx']].subtitle,
            'taskDesc': get_task_name(session['timestep']),
            'body': stages[session['stage_idx']].body,
            'envcaption': stages[session['stage_idx']].envcaption,
        })


def add_stage_to_db(stage_idx, stage_infos, user_seed):
    stage_info = stage_infos[stage_idx]
    stage_info_db.add({
        "stage_idx": stage_idx,
        'stage': utils.encode_json(stages[stage_idx]),
        't': stage_info.t,
        'ep_idx': stage_info.ep_idx,
        'num_success': stage_info.num_success,
        'unique_id': user_seed,
    })
    print('added stage row')


def add_interaction_to_db(socket_json, stage_idx, timestep, rng, user_seed):
    # observation is unnecessarily expensive
    # can regenerate from state information
    #timestep = timestep.replace(observation=None)
    #timestep = serialization.to_state_dict(timestep)
    #timestep = utils.array_to_python(timestep)
    #timestep = json.dumps(timestep)
    new_row = {
        "stage_idx": int(stage_idx),
        "image_seen_time": str(socket_json['imageSeenTime']),
        "key_press_time": str(socket_json['keydownTime']),
        "key": str(socket_json['key']),
        "action": int(keyparser.action(socket_json['key'])),
        # "timestep": timestep,
        "rng": list(rng_from_jax(rng)),
        'unique_id': int(user_seed),
    }


    retry_config = retry.Retry(
        predicate=retry.if_exception_type(
            google.api_core.exceptions.DeadlineExceeded),
        initial=1.0,
        maximum=120.0,  # Increase the maximum retry delay to 120 seconds
        multiplier=2.0,
        deadline=300.0,  # Increase the total time limit for retries to 300 seconds
    )
    @retry.Retry(config=retry_config)
    def add_interaction():
        interactions_db.add(new_row)

    add_interaction()
    print('added interaction row')

#def add_interaction_to_db(socket_json):
#    # observation is unnecessarily expensive
#    # can regenerate from state information
#    timestep = session['timestep']
#    timestep = timestep.replace(observation=None)
#    timestep = serialization.to_state_dict(timestep)
#    timestep = utils.array_to_python(timestep)
#    timestep = json.dumps(timestep)
#    new_row = {
#        "stage_idx": int(session['stage_idx']),
#        "image_seen_time": str(socket_json['imageSeenTime']),
#        "key_press_time": str(socket_json['keydownTime']),
#        "key": str(socket_json['key']),
#        "action": int(keyparser.action(socket_json['key'])),
#        #"timestep": timestep,
#        "rng": list(rng_from_jax(session['rng'])),
#        'unique_id': int(session['user_seed']),
#    }
#    #threading.Thread(target=lambda: interactions_db.add(new_row)).start()
#    interactions_db.add(new_row)
#    print("added new interactions_db row")




#def add_stage_to_db():
#    stage_info = session['stage_infos'][session['stage_idx']]
#    new_row = {
#        "stage_idx": int(session['stage_idx']),
#        'stage': utils.encode_json(stages[session['stage_idx']]),
#        't': int(stage_info.t),
#        'ep_idx': int(stage_info.ep_idx),
#        'num_success': int(stage_info.num_success),
#        'unique_id': int(session['user_seed']),
#    }
#    #threading.Thread(target=lambda: stage_info_db.add(new_row)).start()
#    stage_info_db.add(new_row)
#    print("added new stage_info_db row")

def update_html_fields(**kwargs):
    emit('update_html_fields', {
        'title': stages[session['stage_idx']].title,
        'subtitle': stages[session['stage_idx']].subtitle,
        'body': stages[session['stage_idx']].body,
        'envcaption': stages[session['stage_idx']].envcaption,
        **kwargs,
    })


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
            start_env_interaction_stage()

@socketio.on('record_click')
def handle_record_click(json):
    """Once the experiment has started, the user can go back and forth by clicking arrows.

    This allows for that.
    """
    direction = json['direction']
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
        start_env_interaction_stage()
    else:
        emit('update_content', {
            'content': render_template(template_file)
        })
        update_html_fields()


@socketio.on('key_pressed')
def handle_key_press(json):
    """This happens INSIDE a stage"""

    key = json['key']
    print('key pressed:', key)
    if not keyparser.valid_key(key):
        return

    stage_idx = session['stage_idx']
    stage_info = session['stage_infos'][stage_idx]
    env_params = stages[stage_idx].env_params
    if not session['timestep'].last():
        # update database with image, action, + times of each
        gevent.spawn(add_interaction_to_db, json, stage_idx,
                     session['timestep'], session['rng'], session['user_seed'])


        # take action
        state_image = take_action(key, env_params)
        encoded_image = encode_image(state_image)

        emit('action_taken', {
            'image': encoded_image,
        })

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
            label = f'<span style = "color: {color};" > {label} < /span >'
            update_html_fields(
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
        if key != 'c': return

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
        #------------
        # update to next stage
        # ------------
        if go_to_next_stage:
            gevent.spawn(
                add_stage_to_db, session['stage_idx'], session['stage_infos'], session['user_seed'])
            # update stage idx
            session['stage_idx'] += 1
            session['stage_idx'] = min(session['stage_idx'], len(stages)-1)

            # next template file
            template_file = session.get('template_file', stage.html)

            # update content
            emit('update_content', {
                'content': render_template(template_file),
            })
            update_html_fields()
            if 'env' in template_file:
                start_env_interaction_stage()
            print('advanced to next stage')

        # ------------
        # reset environment
        # ------------
        else:
            state_image = reset_environment(env_params)
            encoded_image = encode_image(state_image)
            update_html_fields(
                taskDesc=get_task_name(session['timestep']),
            )
            emit('action_taken', {
                'image': encoded_image,
            })
            print('reset env')




# Run the Flask app
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8080, debug=True)
