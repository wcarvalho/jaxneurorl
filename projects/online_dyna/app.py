import sys
sys.path.append('ml')

import os

from typing import Callable

from datetime import timedelta
from dotenv import load_dotenv

from flask import Flask
from flask import session
from flask_socketio import emit
from flask_socketio import SocketIO
from flax import serialization, struct
import jax
import json

import numpy as np


import webrl
from webrl import SessionManager
from webrl import StageManager
from webrl import jax_utils

import experiment_1
from experiment_1 import web_env


class DataManager(webrl.DataManager):

    def update_stage_data(
            self,
            stage,
            stage_idx,
            **kwargs):
        def remove_callable(x):
            if isinstance(x, Callable):
                return None
            return x
        stage = jax.tree_map(remove_callable, stage)
        new_row = jax_utils.serialize_pytree(
            stage, jsonify=False)

        # Create the new_row dictionary
        new_row.update({
            "stage_idx": stage_idx,
            'unique_id': int(SessionManager.get('user_seed')),
            **kwargs
        })

        self.stage_data.append(new_row)

    def update_in_episode_data(
            self,
            socket_json: dict,
            stage_idx: int,
            web_env,
            **kwargs):
        """update database with image, action, + times of each."""

        timestep = web_env.timestep
        success = web_env.evaluate_success(timestep)

        timestep = timestep.replace(observation=None)
        timestep = jax_utils.serialize_pytree(timestep, jsonify=False)

        action = web_env.keyparser.action(socket_json.get('key', None))
        action = int(action) if action else action

        new_row = {
            "stage_idx": int(stage_idx),
            "image_seen_time": str(socket_json['imageSeenTime']),
            "key_press_time": str(socket_json['keydownTime']),
            "key": str(socket_json['key']),
            "success": success,
            "action": action,
            "timestep": json.dumps(timestep),
            "rng": list(jax_utils.serialize_rng(SessionManager.get('rng'))),
            'unique_id': int(SessionManager.get('user_seed')),
            # purely for convenience
            **{f"state_{k}": json.dumps(v) for k, v in timestep['state'].items()},
            **kwargs
        }

        self.in_episode_data.append(new_row)



load_dotenv()

DEBUG_APP = int(os.environ.get('DEBUG_APP', 0))
DEBUG_SEED = os.environ.get('DEBUG_SEED', 1)

app = Flask(__name__)
app.secret_key = 'some secret'
app.permanent_session_lifetime = timedelta(days=30)
socketio = SocketIO(app)


index_file = 'index.html'
session_manager = SessionManager(
    index_file=index_file,
    debug=DEBUG_APP
  )

data_manager = DataManager()

stages = experiment_1.stages

stage_manager = StageManager(
    app=app,
    stages=stages,
    data_manager=data_manager,
    web_env=web_env,
    index_file=index_file,
    debug=DEBUG_APP
  )



def message(fn_name: str):
    """PURELY FOR DEBUGGING. Just prints information."""
    print('\n', '='*50, '\n')
    try:
      stage_name = stage_manager.stage.title
      stage_idx = stage_manager.idx
    except:
       stage_idx = 0
       stage_name = 'not set yet'
       
    msg = f"stage_idx: {stage_idx}\n"
    msg += f"fn: {fn_name} | stage: {stage_name}"
    print(msg)
    try:
      emit('get_info', {'info': msg})
    except AttributeError:
       pass



@app.route('/')
def index():
    # boiler-plate. probably don't need to change much.
    message('index')
    SessionManager.set('index_called', True)
    return session_manager.initialize(
        load_user=True,
        initial_template='consent.html',
        experiment_fn_name='experiment',
        )


@app.route('/experiment', methods=['GET', 'POST'])
def experiment(load: bool = False):
    message('experiment')
    #session_manager.experiment_called = True
    SessionManager.set('experiment_called', True)
    if DEBUG_APP:
        rng = jax.random.PRNGKey(DEBUG_SEED)
    else:
        rng = jax.random.PRNGKey(SessionManager.get('user_seed'))

    exp_state = stage_manager.init_state(rng=rng)

    stage_manager.set_state(exp_state)
    return stage_manager.render_stage()


# Register SocketIO events
def register_events():

  @socketio.on('start_connection')
  def start_connection(json: dict = None):
    experiment_initialized = SessionManager.get('experiment_called', False)
    message(f'start_connection, experiment_called = {experiment_initialized}')
    print("input json:", json)
    if experiment_initialized:
      image_data = jax_utils.convert_to_serializable(
         experiment_1.image_data)
      emit('load_data', {'image_data': image_data})
      loaded = stage_manager.maybe_load_state()
      if loaded:
        stage_manager.render_stage_template()

      stage_manager.update_html_content()


  @socketio.on('record_click')
  def handle_record_click(json):
    """Once the experiment has started, the user can go back and forth by clicking arrows.

    This allows for that.
    """
    message('record_click')
    stage_manager.record_click(json=json)

  @socketio.on('key_pressed')
  def handle_key_press(json):
      message('key_pressed')
      stage_manager.handle_key_press(json=json)

  @socketio.on('timer_finished')
  def on_timer_finish():
      message('timer_finished')
      stage_manager.handle_timer_finish()

# Register the events
register_events()

# Run the Flask app
if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0",
                 port=int(os.environ.get("PORT", 8082)), debug=True)
