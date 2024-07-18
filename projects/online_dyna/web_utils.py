import functools
from flask import session
from flask_socketio import emit
from flax import struct
import jax
from typing import Optional, Callable


from housemaze.human_dyna import env as maze

############
# Structures for storing data
############

RenderFn = Callable[[maze.TimeStep, maze.EnvParams, jax.random.PRNGKey], jax.Array]

@struct.dataclass
class Stage:
    html: str
    type: str = 'default'
    title: Optional[str] = ''
    subtitle: Optional[str] = ''
    body: Optional[str] = ''
    envcaption: Optional[str] = ''
    # environment related properties
    seconds: float = None
    render_fn: Optional[RenderFn] = None
    show_progress: bool = True
    show_goal: bool = True
    restart: bool = True
    env_params: Optional[struct.PyTreeNode] = None
    max_episodes: Optional[int] = 10
    min_success: Optional[int] = 1
    count_down_started: bool = False


@struct.dataclass
class StageInfo:
    stage_idx: Stage
    t: int = 0
    ep_idx: int = 0
    num_success: int = 0

############
# Tools for serializing pytrees that include numpy and jax.numpy arrays.
############
"""Tools for serializing pytrees that include numpy and jax.numpy arrays."""
import jax
import jax.numpy as jnp
import json
from json import JSONEncoder
import numpy as np


class CustomJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray, jnp.ndarray)):
            # Convert NumPy and JAX arrays to lists
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64, jnp.float32, jnp.float64)):
            # Convert non-JSON serializable floats to Python floats
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64, jnp.int32, jnp.int64)):
            # Convert non-JSON serializable ints to Python ints
            return int(obj)
        elif hasattr(obj, '__dict__'):
            # Convert custom objects to dictionaries
            return obj.__dict__
        else:
            try:
                # Fallback to default behavior
                return super().default(obj)
            except TypeError as e:
                print(f"Serialization issue with: {obj}, {e}")
                raise


def is_serializable(obj):
    """
    Helper function that returns True if the object can be serialized to JSON, False otherwise.
    """
    try:
        json.dumps(obj, cls=CustomJSONEncoder)
        return True
    except (TypeError, OverflowError):
        return False


def convert_to_serializable(obj):
    if isinstance(obj, (jnp.ndarray, np.ndarray)):
        return obj.tolist()  # Convert JAX array to list
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj



def encode_json(obj):
    encoder = CustomJSONEncoder()
    try:
        return encoder.default(obj)
    except TypeError as e:
        from pprint import pprint
        pprint(
            jax.tree_map(
                lambda x: (is_serializable(x), type(x)), obj))
        import ipdb
        ipdb.set_trace()
        print(f"Cannot serialize {obj} of type {type(obj)}")
        raise e


def array_to_python(obj):
    """Convert JAX objects to Python objects"""
    if isinstance(obj, (jnp.ndarray,)):
        return obj.tolist()
    #elif isinstance(obj, (int, float, str, bool)):
    #    return obj
    elif isinstance(obj, dict):
        return {k: array_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [array_to_python(v) for v in obj]
    #elif hasattr(obj, '__dict__'):
    #    return array_to_python(vars(obj))
    else:
        return obj
