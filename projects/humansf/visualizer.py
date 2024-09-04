import matplotlib.pyplot as plt

import jax
from math import ceil

#from xminigrid.rendering.rgb_render import render as rgb_render
from jaxneurorl.agents.basics import TimeStep
from projects.humansf import keyroom

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional


def display_image(image):
    # Create a new figure and axis
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    # Remove the axis ticks and labels
    ax.axis('off')

    # Show the plot
    plt.tight_layout()
    plt.show()


def init_image(ax, image):
    ax.set_xticks([])
    ax.set_yticks([])
    return ax.imshow(image)


def update_image(im, image):
    im.set_data(image)


class Visualizer(object):
    def __init__(self, env, state_seq, reward_seq=None):
        self.env = env
        self.state_seq = state_seq
        self.reward_seq = reward_seq
        self.fig, self.ax = plt.subplots(1, 1, figsize=(5, 5))
        self.interval = 50

    def animate(
        self,
        save_fname: Optional[str] = "test.gif",
        view: bool = False,
    ):
        """Anim for 2D fct - x (#steps, #pop, 2) & fitness (#steps, #pop)"""
        ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=len(self.state_seq),
            init_func=self.init,
            blit=False,
            interval=self.interval,
        )
        # Save the animation to a gif
        if save_fname is not None:
            ani.save(save_fname)
        # Simply view it 3 times
        if view:
            plt.show(block=False)
            plt.pause(3)
            plt.close()

    def init(self):
        # Plot placeholder points
        self.im = init_image(self.ax, self.state_seq[0])
        self.fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.95])

    def update(self, frame):
        # print(frame)
        update_image(self.im, self.state_seq[frame])
        # self.ax.set_data()
        if self.reward_seq is None:
            self.ax.set_title(
                f"{self.env.name} - Step {frame + 1}", fontsize=15
            )
        else:
            self.ax.set_title(
                "{}: Step {:4.0f} - Return {:7.2f}".format(
                    self.env.name, frame + 1, self.reward_seq[frame]
                ),
                fontsize=15,
            )
def plot_timestep_observations(
        timestep: TimeStep,
        render_fn,
        max_len: int = 40,
        ):

    obs_images = []
    for idx in range(max_len):
        index = lambda y: jax.tree_map(lambda x: x[idx], y)
        #state_image = rgb_render(
        #    grid=timestep.state.grid[idx],
        #    agent=index(timestep.state.agent),
        #    tile_size=tile_size)
        #state_images.append(state_image)
        obs_image = render_fn(index(timestep.state))
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

    fig_width = ncols*width
    fig_height = int(1.7*H*width)
    fig, axs = plt.subplots(
        H, ncols, figsize=(fig_width, fig_height), squeeze=False)

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


    return fig