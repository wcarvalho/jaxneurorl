import functools
from flask import session
from flask_socketio import emit
from flax import struct
import jax
from typing import Optional, List, Callable
from PIL import Image, ImageDraw, ImageFont
from xminigrid.environment import EnvParams
from xminigrid.types import AgentState, TimeStep
from xminigrid.rendering import rgb_render
from xminigrid.core import constants as minigrid_constants
from xminigrid.core import actions as minigrid_actions
import keyroom

############
# Structures for storing data
############

RenderFn = Callable[[TimeStep, EnvParams, jax.random.PRNGKey], jax.Array]
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
fill_coords = rgb_render.fill_coords
point_in_circle = rgb_render.point_in_circle
point_in_triangle = rgb_render.point_in_triangle
point_in_hexagon = rgb_render.point_in_hexagon
point_in_rect = rgb_render.point_in_rect
Colors = minigrid_constants.Colors
COLORS_MAP = {
    Colors.RED: np.array((255, 0, 0)),
    Colors.GREEN: np.array((0, 255, 0)),
    Colors.BLUE: np.array((51, 246, 255)),
    Colors.PURPLE: np.array((240, 51, 255)),
    Colors.YELLOW: np.array((255, 255, 0)),
    Colors.GREY: np.array((222, 218, 222)),
    Colors.BLACK: np.array((0, 0, 0)),
    Colors.ORANGE: np.array((255, 140, 0)),
    Colors.WHITE: np.array((255, 255, 255)),
    Colors.BROWN: np.array((160, 82, 45)),
    Colors.PINK: np.array((225, 20, 147)),
}

def _render_ball(img: np.ndarray, color: int):
    fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS_MAP[color])


def _render_square(img: np.ndarray, color: int):
    fill_coords(img, point_in_rect(0.25, 0.75, 0.25, 0.75), COLORS_MAP[color])


def _render_pyramid(img: np.ndarray, color: int):
    tri_fn = point_in_triangle(
        (0.15, 0.8),
        (0.85, 0.8),
        (0.5, 0.2),
    )
    fill_coords(img, tri_fn, COLORS_MAP[color])


def _render_hex(img: np.ndarray, color: int):
    fill_coords(img, point_in_hexagon(0.35), COLORS_MAP[color])


def _render_star(img: np.ndarray, color: int):
    # yes, this is a hexagram not a star, but who cares...
    tri_fn2 = point_in_triangle(
        (0.15, 0.75),
        (0.85, 0.75),
        (0.5, 0.15),
    )
    tri_fn1 = point_in_triangle(
        (0.15, 0.3),
        (0.85, 0.3),
        (0.5, 0.9),
    )
    fill_coords(img, tri_fn1, COLORS_MAP[color])
    fill_coords(img, tri_fn2, COLORS_MAP[color])


def _render_goal(img: np.ndarray, color: int):
    # draw the grid lines (top and left edges)
    fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
    fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))
    # draw tile
    fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), COLORS_MAP[color])

    # # other grid lines (was used for paper visualizations)
    # fill_coords(img, point_in_rect(1 - 0.031, 1, 0, 1), (100, 100, 100))
    # fill_coords(img, point_in_rect(0, 1, 1 - 0.031, 1), (100, 100, 100))


def _render_key(img: np.ndarray, color: int):
    # Vertical quad
    fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), COLORS_MAP[color])
    # Teeth
    fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), COLORS_MAP[color])
    fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), COLORS_MAP[color])
    # Ring
    fill_coords(img, point_in_circle(
        cx=0.56, cy=0.28, r=0.190), COLORS_MAP[color])
    fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))


def _render_door_locked(img: np.ndarray, color: int):
    fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), COLORS_MAP[color])
    fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94),
                0.45 * COLORS_MAP[color])
    # Draw key slot
    fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), COLORS_MAP[color])


def _render_door_closed(img: np.ndarray, color: int):
    fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), COLORS_MAP[color])
    fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
    fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), COLORS_MAP[color])
    fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))
    # Draw door handle
    fill_coords(img, point_in_circle(
        cx=0.75, cy=0.50, r=0.08), COLORS_MAP[color])


def _render_door_open(img: np.ndarray, color: int):
    rgb_render._render_floor(img, Colors.BLACK)
    # draw the grid lines (top and left edges)
    fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
    fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))
    # draw door
    fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), COLORS_MAP[color])
    fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))

POCKET_FN_MAP = {
    minigrid_constants.Tiles.BALL: _render_ball,
    minigrid_constants.Tiles.SQUARE: _render_square,
    minigrid_constants.Tiles.PYRAMID: _render_pyramid,
    minigrid_constants.Tiles.HEX: _render_hex,
    minigrid_constants.Tiles.STAR: _render_star,
    minigrid_constants.Tiles.GOAL: _render_goal,
    minigrid_constants.Tiles.KEY: _render_key,
}

TILES_FN_MAP = {
    minigrid_constants.Tiles.FLOOR: rgb_render._render_floor,
    minigrid_constants.Tiles.WALL: rgb_render._render_wall,
    minigrid_constants.Tiles.DOOR_LOCKED: _render_door_locked,
    minigrid_constants.Tiles.DOOR_CLOSED: _render_door_closed,
    minigrid_constants.Tiles.DOOR_OPEN: _render_door_open,
    minigrid_constants.Tiles.EMPTY: lambda img, color: img,
    **POCKET_FN_MAP,
}


@functools.cache
def render_tile(
    tile: np.ndarray,
    agent_direction: int = None,
    agent_pocket = None,
    highlight: bool = False, tile_size: int = 32, subdivs: int = 3
) -> np.ndarray:
    img = np.full((tile_size * subdivs, tile_size * subdivs, 3),
                  dtype=np.uint8, fill_value=255)

    # draw tile
    rgb_render._render_floor(img, Colors.BLACK)
    TILES_FN_MAP[tile[0]](img, tile[1])

    # draw agent if on this tile
    if agent_direction is not None:
        rgb_render._render_player(img, agent_direction)

    if agent_pocket is not None:
        if agent_pocket is not (minigrid_constants.Tiles.EMPTY,
                            minigrid_constants.Colors.EMPTY):
            if agent_pocket[0] in POCKET_FN_MAP.keys():
                POCKET_FN_MAP[agent_pocket[0]](img, agent_pocket[1])


    if highlight:
        rgb_render.highlight_img(img, alpha=0.2)

    # downsample the image to perform supersampling/anti-aliasing
    img = rgb_render.downsample(img, subdivs)

    return img

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
                agent_pocket = tuple(np.asarray(agent.pocket))
            else:
                agent_direction = None
                agent_pocket = None

            tile_img = render_tile(
                tile=tuple(grid[y, x]),
                agent_direction=agent_direction,
                agent_pocket=agent_pocket,
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


def render_map(timestep, env_params=None, rng=None):
    return render(
        np.asarray(timestep.state.grid),
        timestep.state.agent,
        0,
        tile_size=32)


def render_keys(timestep, env_params, rng):
    objects = env_params.maze_config['keys']
    objects = permute(objects, rng)
    return objects_with_number(objects)

def render_pairs(timestep, env_params, rng):
    objects = env_params.maze_config['pairs'].reshape(-1, 2)
    objects = permute(objects, rng)
    return objects_with_number(objects)


class KeyRoomUpDownLeftRight(keyroom.KeyRoom):

    def take_action(self, key, timestep, action, params):
        del key
        del params
        new_grid, new_agent, _ = take_action(
            timestep.state.grid, timestep.state.agent, action)

        new_grid, new_agent = self.teleport_agent_remove_key_close_door(
            prior_timestep=timestep,
            new_grid=new_grid,
            new_agent=new_agent)

        return new_grid, new_agent, _


def take_action(grid, agent, action):
    # This will evaluate all actions.
    # Can we fix this and choose only one function? It'll speed everything up dramatically.
    def move(grid, agent, direction):
        agent = agent.replace(direction=direction)
        grid, agent, pos = minigrid_actions.move_forward(grid, agent)
        return grid, agent, pos

    def interact(grid, agent):
        grid, agent, pos = minigrid_actions.toggle(grid, agent)
        grid, agent, pos = minigrid_actions.pick_up(grid, agent)
        return grid, agent, pos

    actions = (
        lambda: move(grid, agent, 0),  # up
        lambda: move(grid, agent, 1),  # right
        lambda: move(grid, agent, 2),  # down
        lambda: move(grid, agent, 3),  # left
        lambda: interact(grid, agent),
        lambda: minigrid_actions.put_down(grid, agent),
    )
    new_grid, new_agent, changed_position = jax.lax.switch(action, actions)

    return new_grid, new_agent, changed_position

############
# Flask functions
############
