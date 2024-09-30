import numpy as np

import jax
import jax.tree_util as jtu
import jax.numpy as jnp
import pdb

from housemaze import levels
from housemaze import env as maze
from housemaze import renderer
from housemaze import utils

maze1 = """
.....
..>..
.....
A....
.....
""".strip()

maze2 = """
..B..
.....
.....
...<.
.....
""".strip()

maze3 = """
.....
.....
..C..
.....
.>...
""".strip()

maze4 = """
...>.
.....
.....
.....
.D...
""".strip()

maze5 = """
.....
.....
...<E
.....
.....
""".strip()

def housemaze_env():
    char_to_key=dict(
        A="knife",
        B="fork",
        C="pan",
        D="pot",
        E="bowl",
        F="plates",
    )

    mazes = [maze1, maze2, maze3, maze4, maze5]

    image_dict = utils.load_image_dict()

    object_to_index = {key: idx for idx, key in enumerate(image_dict['keys'])}

    objects = np.array([object_to_index[v] for v in char_to_key.values()])

    map_inits = [utils.from_str(maze,
                                char_to_key=char_to_key,
                                object_to_index=object_to_index)
                 for maze in mazes]

    map_init = jtu.tree_map(lambda *v: jnp.stack(v), *map_inits)
    map_init = maze.MapInit(*map_init)
    # create env params
    env_params = maze.EnvParams(
        map_init=jax.tree_map(jnp.asarray, map_init),
        time_limit=jnp.array(50),
        objects=jnp.asarray(objects))

    seed = 6
    rng = jax.random.PRNGKey(seed)

    task_runner = maze.TaskRunner(task_objects=env_params.objects)
    env = maze.HouseMaze(
        task_runner=task_runner,
        num_categories=len(image_dict['keys']),
    )
    env = utils.AutoResetWrapper(env)

    return env_params, env