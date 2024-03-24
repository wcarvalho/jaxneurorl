import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional
from gymnax.visualize.vis_gym import init_gym, update_gym
from gymnax.visualize.vis_minatar import init_minatar, update_minatar
from gymnax.visualize.vis_circle import init_circle, update_circle
from gymnax.visualize.vis_maze import init_maze, update_maze
from gymnax.visualize.vis_catch import init_catch, update_catch


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
    def __init__(self, env, env_params, state_seq, reward_seq=None):
        self.env = env
        self.env_params = env_params
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


if __name__ == "__main__":
    import jax
    import jax.numpy as jnp
    import gymnax

    rng = jax.random.PRNGKey(0)
    env, env_params = gymnax.make("Pong-misc")

    state_seq, reward_seq = [], []
    rng, rng_reset = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)
    while True:
        state_seq.append(env_state)
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        action = env.action_space(env_params).sample(rng_act)
        next_obs, next_env_state, reward, done, info = env.step(
            rng_step, env_state, action, env_params
        )
        reward_seq.append(reward)
        if done:
            break
        else:
            obs = next_obs
            env_state = next_env_state

    cum_rewards = jnp.cumsum(jnp.array(reward_seq))
    vis = Visualizer(env, env_params, state_seq, cum_rewards)
    vis.animate(f"anim.gif")