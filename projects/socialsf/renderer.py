import numpy as np
import jax
import jax.numpy as jnp
import chex
import matplotlib.pyplot as plt

from functools import partial


class MazeRenderer(object):
  """This class renders the maze for visual logging, compatible with jit.

  Args:
      env (Maze):
      tile_size (int, optional): The number of pixels each tile should take up. Defaults to 32.
      render_border (bool, optional): If true, renders the one-tile thick border around the level. Defaults to True.
  """

  def __init__(self, env: Maze, tile_size: int = 32, render_border: bool = True):
    self.env = env
    self.tile_size = tile_size
    self.render_border = render_border
    self._atlas = jnp.array(_make_tile_atlas(tile_size))

  @partial(jax.jit, static_argnums=(0,))
  def render_level(self, level, env_params):
    # For Minigrid, env_state contains all attributes of level,
    # and only uses these attributes. So can just call render_state.
    # However, in general, these routines may be a bit different.
    # For example, levels may map to many different start states.
    # As such, one may want to render an image representative of all
    # possible start states when rendering a level.
    return self.render_state(level, env_params)

  @partial(jax.jit, static_argnums=(0,))
  def render_state(
    self, env_state: EnvState, env_params: EnvParams, highlight=False
  ) -> chex.Array:
    tile_size = self.tile_size
    nrows = self.env.max_height + 2 * self.render_border
    ncols = self.env.max_width + 2 * self.render_border
    width_px = ncols * tile_size
    height_px = nrows * tile_size

    agent_pos = env_state.agent_pos + self.render_border
    goal_pos = env_state.goal_pos + self.render_border

    if self.render_border:
      cells = jnp.pad(
        jnp.where(env_state.wall_map, 1, 0),
        1,
        mode="constant",
        constant_values=True,
      )
    else:
      cells = jnp.where(env_state.wall_map, 1, 0)
    cells = jnp.where(cells, 1, 0)
    cells = cells.at[agent_pos[1], agent_pos[0]].set(3 + env_state.agent_dir)
    cells = cells.at[goal_pos[1], goal_pos[0]].set(2)

    img = self._atlas[cells].transpose(0, 2, 1, 3, 4).reshape(height_px, width_px, 3)

    f_vec = DIR_TO_VEC[env_state.agent_dir]
    r_vec = jnp.array([-f_vec[1], f_vec[0]])

    agent_view_size = self.env.agent_view_size

    min_bound = jnp.min(
      jnp.stack(
        [
          agent_pos,
          agent_pos + f_vec * (agent_view_size - 1),
          agent_pos - r_vec * (agent_view_size // 2),
          agent_pos + r_vec * (agent_view_size // 2),
        ]
      ),
      0,
    )

    min_x = jnp.minimum(
      jnp.maximum(min_bound[0], 0),
      env_state.wall_map.shape[0] - 1 + 2 * self.render_border,
    )
    min_y = jnp.minimum(
      jnp.maximum(min_bound[1], 0),
      env_state.wall_map.shape[1] - 1 + 2 * self.render_border,
    )
    max_x = jnp.minimum(
      jnp.maximum(min_bound[0] + agent_view_size, 0),
      env_state.wall_map.shape[0] + 2 * self.render_border,
    )
    max_y = jnp.minimum(
      jnp.maximum(min_bound[1] + agent_view_size, 0),
      env_state.wall_map.shape[1] + 2 * self.render_border,
    )

    all_pos = jnp.arange(ncols * nrows)
    mask = (
      ((all_pos % ncols) >= min_x)
      & ((all_pos % ncols) < max_x)
      & ((all_pos // ncols) >= min_y)
      & ((all_pos // ncols) < max_y)
    )
    mask = jnp.kron(
      mask.reshape(nrows, ncols), jnp.ones((self.tile_size, self.tile_size))
    )[..., None]

    if highlight:
      highlight_img = (img + 0.3 * (255 - img)).astype(jnp.uint8).clip(0, 255)
    else:
      highlight_img = img
    return jnp.where(mask, highlight_img, img)


def _make_tile_atlas(tile_size):
  TRI_COORDS = np.array(
    [
      [0.12, 0.19],
      [0.87, 0.50],
      [0.12, 0.81],
    ]
  )

  def fill_coords(img, fn, color):
    new_img = img.copy()
    for y in range(img.shape[0]):
      for x in range(img.shape[1]):
        yf = (y + 0.5) / img.shape[0]
        xf = (x + 0.5) / img.shape[1]
        if fn(xf, yf):
          new_img[y, x] = color
    return new_img

  def point_in_rect(xmin, xmax, ymin, ymax):
    def fn(x, y):
      return x >= xmin and x <= xmax and y >= ymin and y <= ymax

    return fn

  def point_in_triangle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    def fn(x, y):
      v0 = c - a
      v1 = b - a
      v2 = np.array((x, y)) - a

      # Compute dot products
      dot00 = np.dot(v0, v0)
      dot01 = np.dot(v0, v1)
      dot02 = np.dot(v0, v2)
      dot11 = np.dot(v1, v1)
      dot12 = np.dot(v1, v2)

      # Compute barycentric coordinates
      inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
      u = (dot11 * dot02 - dot01 * dot12) * inv_denom
      v = (dot00 * dot12 - dot01 * dot02) * inv_denom

      # Check if point is in triangle
      return (u >= 0) and (v >= 0) and (u + v) < 1

    return fn

  atlas = np.empty((7, tile_size, tile_size, 3), dtype=np.uint8)

  def add_border(tile):
    new_tile = fill_coords(tile, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
    return fill_coords(new_tile, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

  atlas[0] = add_border(np.tile([0, 0, 0], (tile_size, tile_size, 1)))  # empty
  atlas[1] = np.tile([100, 100, 100], (tile_size, tile_size, 1))  # wall
  atlas[2] = np.tile([0, 255, 0], (tile_size, tile_size, 1))  # goal

  # Handle player
  agent_tile = np.tile([0, 0, 0], (tile_size, tile_size, 1))
  agent_tile = fill_coords(agent_tile, point_in_triangle(*TRI_COORDS), [255, 0, 0])

  atlas[3] = add_border(agent_tile)  # right
  atlas[4] = add_border(np.rot90(agent_tile, k=3))  # down
  atlas[5] = add_border(np.rot90(agent_tile, k=2))  # left
  atlas[6] = add_border(np.rot90(agent_tile, k=1))  # up

  return atlas


def place_arrows_on_image(image, actions, env_state, offset=8, ax=None):
  # Get the dimensions of the image and the maze
  image_height, image_width, _ = image.shape
  maze_height, maze_width, _ = env_state.maze_map.shape

  # Calculate the scaling factors for mapping maze coordinates to image coordinates
  scale_y = (image_height - offset) // maze_height
  scale_x = (image_width - offset) // maze_width

  # Calculate the offset to account for the border of walls
  offset_y = (image_height - scale_y * maze_height) // 2
  offset_x = (image_width - scale_x * maze_width) // 2

  # Create a figure and axis
  if ax is None:
    fig, ax = plt.subplots(1, figsize=(8, 8))

  # Display the rendered image
  ax.imshow(image)

  # Iterate over each position in the maze
  for (y, x), action in actions.items():
    # Calculate the center coordinates of the cell in the image
    center_y = offset_y + (y + 0.5) * scale_y
    center_x = offset_x + (x + 0.5) * scale_x

    # Define the arrow directions based on the action
    if action == "up":
      dx, dy = 0, -scale_y / 2
    elif action == "down":
      dx, dy = 0, scale_y / 2
    elif action == "left":
      dx, dy = -scale_x / 2, 0
    elif action == "right":
      dx, dy = scale_x / 2, 0

    # Draw the arrow on the image
    ax.arrow(
      center_x,
      center_y,
      dx,
      dy,
      head_width=scale_x / 10,
      head_length=scale_y / 10,
      fc="g",
      ec="g",
    )

  # Remove the axis ticks and labels
  ax.set_xticks([])
  ax.set_yticks([])
  return ax


def show_path_to_goal(image, start_pos, actions, env_state, offset=8, ax=None):
  # Get the dimensions of the image and the maze
  image_height, image_width, _ = image.shape
  maze_height, maze_width, _ = env_state.maze_map.shape

  # Calculate the scaling factors for mapping maze coordinates to image coordinates
  scale_y = (image_height - offset) // maze_height
  scale_x = (image_width - offset) // maze_width

  # Calculate the offset to account for the border of walls
  offset_y = (image_height - scale_y * maze_height) // 2
  offset_x = (image_width - scale_x * maze_width) // 2

  # Create a figure and axis if not provided
  if ax is None:
    fig, ax = plt.subplots(1, figsize=(8, 8))

  # Display the rendered image
  ax.imshow(image)

  # # Add grid lines
  # for i in range(maze_height + 1):
  #     ax.axhline(offset_y + i * scale_y, color='w', linewidth=0.5)
  # for j in range(maze_width + 1):
  #     ax.axvline(offset_x + j * scale_x, color='w', linewidth=0.5)

  # Initialize the current position as the starting position
  current_pos = (start_pos[1], start_pos[0])

  # Iterate until the goal is reached
  while current_pos in actions:
    action = actions[current_pos]

    # Calculate the center coordinates of the cell in the image
    y, x = current_pos
    center_y = offset_y + (y + 0.5) * scale_y
    center_x = offset_x + (x + 0.5) * scale_x

    # Define the arrow directions based on the action
    if action == "up":
      dx, dy = 0, -scale_y / 2
    elif action == "down":
      dx, dy = 0, scale_y / 2
    elif action == "left":
      dx, dy = -scale_x / 2, 0
    elif action == "right":
      dx, dy = scale_x / 2, 0

    # Draw the arrow on the image
    ax.arrow(
      center_x,
      center_y,
      dx,
      dy,
      head_width=scale_x / 10,
      head_length=scale_y / 8,
      fc="g",
      ec="g",
    )

    # Update the current position based on the action
    if action == "up":
      current_pos = (y - 1, x)
    elif action == "down":
      current_pos = (y + 1, x)
    elif action == "left":
      current_pos = (y, x - 1)
    elif action == "right":
      current_pos = (y, x + 1)

  # Remove the axis ticks and labels
  ax.set_xticks([])
  ax.set_yticks([])

  return ax
