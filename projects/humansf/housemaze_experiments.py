
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import os.path

from housemaze import levels
from housemaze.human_dyna import utils as housemaze_utils
from housemaze.human_dyna import env as maze
from housemaze.human_dyna import mazes

#def load_env_params(
#      num_groups: int,
#      max_objects: int = 3,
#      file: str = None,
#      large_only: bool = False,
#    ):
#    # load groups
#    list_of_groups = housemaze_utils.load_groups(file)

#    group_set = list_of_groups[0]
#    assert num_groups <= 3
#    group_set = group_set[:num_groups]

#    # load levels
#    pretrain_level = levels.two_objects
#    train_level = levels.three_pairs_maze1

#    ##################
#    # create reset parameters
#    ##################
#    make_int_array = lambda x: jnp.asarray(x, dtype=jnp.int32)
#    def make_reset_params(
#        map_init,
#        train_objects,
#        test_objects,
#        **kwargs):

#      train_objects_ = np.ones(max_objects)*-1
#      train_objects_[:len(train_objects)] = train_objects
#      test_objects_ = np.ones(max_objects)*-1
#      test_objects_[:len(test_objects)] = test_objects
#      map_init = map_init.replace(
#          grid=make_int_array(map_init.grid),
#          agent_pos=make_int_array(map_init.agent_pos),
#          agent_dir=make_int_array(map_init.agent_dir),
#      )
#      return maze.ResetParams(
#          map_init=map_init,
#          train_objects=make_int_array(train_objects_),
#          test_objects=make_int_array(test_objects_),
#          **kwargs,
#      )
       
#    list_of_reset_params = []
#    num_starting_locs = 4
#    max_starting_locs = 10
#    # -------------
#    # pretraining levels
#    # -------------
#    for group in group_set:
#      list_of_reset_params.append(
#          make_reset_params(
#              map_init=maze.MapInit(*housemaze_utils.from_str(
#                  pretrain_level, char_to_key=dict(A=group[0], B=group[1]))),
#              train_objects=group[:1],
#              test_objects=group[1:],
#              label=jnp.array(1),
#              starting_locs=make_int_array(
#                  np.ones((len(group_set), max_starting_locs, 2))*-1)
#          )
#      )

#    # -------------
#    # MAIN training level
#    # -------------
#    if large_only:
#       list_of_reset_params = []
#    train_objects = group_set[:, 0]
#    test_objects = group_set[:, 1]
#    map_init = maze.MapInit(*housemaze_utils.from_str(
#        train_level,
#        char_to_key=dict(
#            A=group_set[0, 0],
#            B=group_set[0, 1],
#            C=group_set[1, 0],
#            D=group_set[1, 1],
#            E=group_set[2, 0],
#            F=group_set[2, 1],
#        )))

#    all_starting_locs = np.ones((len(group_set), max_starting_locs, 2))*-1
#    for idx, goal in enumerate(train_objects):
#        path = housemaze_utils.find_optimal_path(
#            map_init.grid, map_init.agent_pos, np.array([goal]))
#        width = len(path)//num_starting_locs
#        starting_locs = np.array([path[i] for i in range(0, len(path), width)])
#        all_starting_locs[idx, :len(starting_locs)] = starting_locs

#    list_of_reset_params.append(
#        make_reset_params(
#            map_init=map_init,
#            train_objects=train_objects,
#            test_objects=test_objects,
#            starting_locs=make_int_array(all_starting_locs),
#            curriculum=jnp.array(True),
#            label=jnp.array(0),
#        )
#    )

#    return group_set, maze.EnvParams(
#        reset_params=jtu.tree_map(
#            lambda *v: jnp.stack(v), *list_of_reset_params),
#    )


def get_group_set(num_groups):
    list_of_groups = housemaze_utils.load_groups()
    full_group_set = list_of_groups[0]

    chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    char2key = dict()
    for idx, char in enumerate(chars):
        i, j = idx // 2, idx % 2
        if i > len(full_group_set):
            break
        char2key[char] = full_group_set[i, j]

    assert num_groups <= 3
    task_group_set = full_group_set[:num_groups]
    task_objects = task_group_set.reshape(-1)

    return char2key, task_group_set, task_objects


def maze1_all(config):
    """Maze 1: testing offtaskness for all 3 spaces."""
    num_groups = config['rlenv']['ENV_KWARGS'].pop('NUM_GROUPS', 3)
    char2key, group_set, task_objects = get_group_set(num_groups)

    pretrain_params = mazes.get_pretraining_reset_params(
        group_set=group_set,
    )
    main_params = mazes.get_maze_reset_params(
        group_set=group_set,
        char2key=char2key,
        maze_str=mazes.maze1,
        curriculum=True,
    )
    train_params = maze.EnvParams(
        reset_params=jtu.tree_map(
            lambda *v: jnp.stack(v), *(pretrain_params + main_params)),
    )

    test_params = maze.EnvParams(
        reset_params=jtu.tree_map(
            lambda *v: jnp.stack(v), *main_params),
    ).replace(training=False)

    return train_params, test_params, task_objects


def maze3_open(config):
    """Maze 3: testing if open space is skipped. should be."""
    num_groups = config['rlenv']['ENV_KWARGS'].pop('NUM_GROUPS', 1)
    char2key, group_set, task_objects = get_group_set(num_groups)

    pretrain_params = mazes.get_pretraining_reset_params(
        group_set=group_set,
    )
    main_params = mazes.get_maze_reset_params(
        group_set=group_set,
        char2key=char2key,
        maze_str=mazes.maze3,
        label=jnp.array(0),
        curriculum=True,
    )
    main_open_params = mazes.get_maze_reset_params(
        group_set=group_set,
        char2key=char2key,
        maze_str=mazes.maze3_open,
        label=jnp.array(2),
    )

    train_params = pretrain_params + main_params
    train_params = maze.EnvParams(
        reset_params=jtu.tree_map(
            lambda *v: jnp.stack(v), *train_params),
    )

    test_params = main_params + main_open_params
    test_params = maze.EnvParams(
        reset_params=jtu.tree_map(
            lambda *v: jnp.stack(v), *test_params),
    ).replace(training=False)

    return train_params, test_params, task_objects


def maze3_randomize(config):
    """Maze 3: testing if open space is skipped. should be."""
    num_groups = config['rlenv']['ENV_KWARGS'].pop('NUM_GROUPS', 1)
    char2key, group_set, task_objects = get_group_set(num_groups)

    pretrain_params = mazes.get_pretraining_reset_params(
        group_set=group_set,
    )
    main_params = mazes.get_maze_reset_params(
        group_set=group_set,
        char2key=char2key,
        maze_str=mazes.maze3,
    )
    main_open_params = mazes.get_maze_reset_params(
        group_set=group_set,
        char2key=char2key,
        maze_str=mazes.maze3_open,
    )

    train_params = pretrain_params + main_params
    train_params = maze.EnvParams(
        randomize_agent=True,
        reset_params=jtu.tree_map(
            lambda *v: jnp.stack(v), *train_params),
    )

    test_params = main_params + main_open_params
    test_params = maze.EnvParams(
        training=False,
        reset_params=jtu.tree_map(
            lambda *v: jnp.stack(v), *test_params),
    )

    return train_params, test_params, task_objects
