import matplotlib.pyplot as plt

import jax
from math import ceil

#from xminigrid.rendering.rgb_render import render as rgb_render
from singleagent.basics import TimeStep
from projects.humansf import keyroom

def plot_timestep_observations(
        timestep: TimeStep,
        max_len: int = 40,
        tile_size: int =8):

    obs_images = []
    for idx in range(max_len):
        index = lambda y: jax.tree_map(lambda x: x[idx], y)
        #state_image = rgb_render(
        #    grid=timestep.state.grid[idx],
        #    agent=index(timestep.state.agent),
        #    tile_size=tile_size)
        #state_images.append(state_image)
        obs_image = keyroom.render_room(
            index(timestep.state),
            tile_size=tile_size)
        obs_images.append(obs_image)


def plot_frames(
        timesteps,
        frames,
        ncols,
        panel_title_fn = lambda t,i: '',
        max_frames=1e10):
    """
    Dynamically plots frames in a single figure based on the number of columns (W) and maximum number of frames.

    :param task_name: Name of the task to be displayed as figure title.
    :param frames: 4D numpy array of shape (T, H, W, C) containing the frames to plot.
    :param rewards: List of rewards corresponding to actions taken.
    :param actions_taken: List of actions taken corresponding to each frame.
    :param max_frames: Maximum number of frames to plot.
    :param W: Number of columns in the plot's grid.
    """
    T = min(len(frames), max_frames)  # Total number of frames to plot (limited by max_frames)
    H = ceil(T / ncols)  # Calculate number of rows required
    width = 3

    fig, axs = plt.subplots(H, ncols, figsize=(ncols*width, H*width), squeeze=False)

    # Flatten the axes array for easy iteration
    axs = axs.ravel()

    for i in range(T):
        ax = axs[i]
        ax.imshow(frames[i])
        ax.axis('off')  # Hide the axis
        ax.set_title(panel_title_fn(timesteps, i))

    # Hide unused subplots
    for i in range(T, H * ncols):
        axs[i].axis('off')

    plt.tight_layout()

    return fig