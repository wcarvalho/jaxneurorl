from flask import session
from flask_socketio import emit
from xminigrid.types import AgentState
from xminigrid.rendering.rgb_render import render_tile
from typing import Optional

from flax import struct


############
# Structures for storing data
############

@struct.dataclass
class Stage:
    html: str
    title: Optional[str] = ''
    subtitle: Optional[str] = ''
    body: Optional[str] = ''
    envcaption: Optional[str] = ''
    env_params: Optional[struct.PyTreeNode] = None
    max_episodes: Optional[int] = 10
    min_success: Optional[int] = 1


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




################
# Env utils
################

def render(
    grid: np.ndarray,
    agent: AgentState | None = None,
    view_size: int = 7,
    tile_size: int = 32,
) -> np.ndarray:
    # grid = np.asarray(grid)
    # compute the total grid size
    height_px = grid.shape[0] * int(tile_size)
    width_px = grid.shape[1] * int(tile_size)

    img = np.full((height_px, width_px, 3), dtype=np.uint8, fill_value=-1)

    # compute agent fov highlighting

    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if agent is not None and np.array_equal((y, x), agent.position):
                agent_direction = int(agent.direction)
            else:
                agent_direction = None

            tile_img = render_tile(
                tile=tuple(grid[y, x]),
                agent_direction=agent_direction,
                tile_size=int(tile_size),
            )

            ymin = y * int(tile_size)
            ymax = (y + 1) * int(tile_size)
            xmin = x * int(tile_size)
            xmax = (x + 1) * int(tile_size)
            img[ymin:ymax, xmin:xmax, :] = tile_img

    return img

############
# Flask functions
############
