from typing import Callable, Optional, Any

from library import loggers
from projects.humansf import observers as humansf_observers


from gymnax.environments import environment


import functools
from typing import Callable


def make_logger(
        config: dict,
        env: environment.Environment,
        env_params: environment.EnvParams,
        maze_config: dict,
        action_names: dict,
        get_task_name: Callable = None,
        learner_log_extra: Optional[Callable[[Any], Any]] = None
        ):
    return loggers.Logger(
        gradient_logger=loggers.default_gradient_logger,
        learner_logger=loggers.default_learner_logger,
        experience_logger=functools.partial(
            humansf_observers.experience_logger,
            action_names=action_names,
            get_task_name=get_task_name,
            max_len=config['MAX_EPISODE_LOG_LEN'],
            ),
        learner_log_extra=learner_log_extra,
    )