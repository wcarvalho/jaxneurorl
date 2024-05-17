from flask import session
from flask_socketio import emit
from flax import struct
from typing import Optional, List
from PIL import Image, ImageDraw, ImageFont
from xminigrid.types import AgentState
from xminigrid.rendering.rgb_render import render_tile


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


def permute(x: jax.Array, rng: jax.random.PRNGKey):
    # Generate random permutation indices
    permutation = jax.random.permutation(rng, x.shape[0])

    # Apply permutation to the array
    return x[permutation]

def render_object_with_number(
        object: np.ndarray,
        number: int,
        image_width: int=50, font_size: int=20):
    assert object.ndim == 1, 'must have 1 dimension'
    assert object.shape[0] == 2, 'render_tile only supports rendering [shape, color]'

    image_height = 2*image_width  # for image + number
    image_size = (image_width, image_height)
    # Create the key image
    key_image = render_tile(tuple(object))

    # Ensure the key image is an instance of a PIL Image
    key_image = Image.fromarray(np.array(key_image, dtype=np.uint8))

    # Create a blank white image for the combined key and number
    combined_image = Image.new('RGB', image_size, color=(255, 255, 255))

    # Resize the key image to fit within the combined image
    key_image = key_image.resize((image_size[0], image_size[0]))

    # Paste the key image onto the combined image
    combined_image.paste(key_image, (0, 0))

    # Initialize the drawing context
    draw = ImageDraw.Draw(combined_image)

    # Load a built-in PIL font
    font = ImageFont.load_default(size=font_size)

    # Draw the number beneath the key
    text = str(number)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_position = ((image_size[0] - text_width) // 2, image_size[0] +
                    (image_size[1] - image_size[0] - text_height) // 2)

    draw.text(text_position, text, fill=(0, 0, 0), font=font)

    # Convert to NumPy array
    np_image = np.array(combined_image)

    return np_image

def objects_with_number(
        objects,
        rng: Optional[jax.random.PRNGKey] = None,
        numbers: List[int] = None):

    if isinstance(objects, jnp.ndarray):
        objects = np.asarray(objects)

    if numbers is None:
        numbers = [i+1 for i in range(len(objects))]

    key_images_with_numbers = [
        render_object_with_number(o, n) for (o,n) in zip(objects, numbers)]

    # Combine the images into one
    return np.hstack(key_images_with_numbers)


############
# Flask functions
############
