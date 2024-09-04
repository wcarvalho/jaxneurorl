
import copy
from gymnax.environments import environment
import jax
import jax.numpy as jnp

import flax
from flax.core import FrozenDict
from flax.training.train_state import TrainState

import flashbax as fbx
from safetensors.flax import save_file
from flax.traverse_util import flatten_dict
from agents import value_based_basics as vbb

from library import observers
from library import loggers


class CustomTrainState(TrainState):
    target_network_params: flax.core.FrozenDict
    timesteps: int
    n_updates: int
    n_logs: int
    n_extra_updates: int = 0


def save_params(params, n_updates, filename_fn):
    def callback(p, n):
        filename = filename_fn(n)
        flattened_dict = flatten_dict(p, sep=',')
        save_file(flattened_dict, filename)
        print(f'Saved: {filename}')

    jax.debug.callback(callback, params, n_updates)



def make_train(
        config: dict,
        env: environment.Environment,
        train_env_params: environment.EnvParams,
        make_agent: vbb.MakeAgentFn,
        make_optimizer: vbb.MakeOptimizerFn,
        make_loss_fn_class: vbb.MakeLossFnClass,
        make_replay_loss_fn_class: vbb.MakeLossFnClass,
        make_actor: vbb.MakeActorFn,
        make_logger: vbb.MakeLoggerFn = loggers.default_make_logger,
        test_env_params: vbb.Optional[environment.EnvParams] = None,
        save_params_fn=save_params,
        ObserverCls: observers.BasicObserver = observers.BasicObserver,
):
    """Creates a train function that does learning after unrolling agent for K timesteps.

    Args:
        config (dict): _description_
        env (environment.Environment): _description_
        env_params (environment.EnvParams): _description_
        make_agent (MakeAgentFn): _description_
        make_optimizer (MakeOptimizerFn): _description_
        make_loss_fn_class (MakeLossFnClass): _description_
        make_actor (MakeActorFn): _description_
        test_env_params (Optional[environment.EnvParams], optional): _description_. Defaults to None.
        ObserverCls (observers.BasicObserver, optional): _description_. Defaults to observers.BasicObserver.

    Returns:
        _type_: _description_
    """

    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_ENVS"] // config["TRAINING_INTERVAL"]
    )
    test_env_params = test_env_params or copy.deepcopy(train_env_params)

    def vmap_reset(rng, env_params):
      return jax.vmap(env.reset, in_axes=(0, None))(
          jax.random.split(rng, config["NUM_ENVS"]), env_params)

    def vmap_step(rng, env_state, action, env_params):
       return jax.vmap(
           env.step, in_axes=(0, 0, 0, None))(
           jax.random.split(rng, config["NUM_ENVS"]), env_state, action, env_params)

    def train(rng: jax.random.PRNGKey):
        logger = make_logger(config, env, train_env_params)

        ##############################
        # INIT ENV
        ##############################
        rng, _rng = jax.random.split(rng)
        init_timestep = vmap_reset(rng=_rng, env_params=train_env_params)

        ##############################
        # INIT NETWORK
        # will be absorbed into _update_step via closure
        ##############################
        rng, _rng = jax.random.split(rng)
        agent, network_params, agent_reset_fn = make_agent(
            config=config,
            env=env,
            env_params=train_env_params,
            example_timestep=init_timestep,
            rng=_rng)

        vbb.log_params(network_params['params'])

        rng, _rng = jax.random.split(rng)
        init_agent_state = agent_reset_fn(network_params, init_timestep, _rng)

        ##############################
        # INIT Actor
        # will be absorbed into _update_step via closure
        ##############################
        rng, _rng = jax.random.split(rng)
        actor = make_actor(
            config=config,
            agent=agent,
            rng=_rng)

        ##############################
        # INIT OPTIMIZER
        ##############################
        tx = make_optimizer(config)

        train_state = CustomTrainState.create(
            apply_fn=agent.apply,
            params=network_params,
            target_network_params=jax.tree_map(
                lambda x: jnp.copy(x), network_params),
            tx=tx,
            timesteps=0,
            n_updates=0,
            n_logs=0,
        )

        ##############################
        # INIT BUFFER
        ##############################
        period = config.get("SAMPLING_PERIOD", 1)
        total_batch_size = config.get("TOTAL_BATCH_SIZE")
        sample_batch_size = config['BUFFER_BATCH_SIZE']
        sample_sequence_length = config.get('SAMPLE_LENGTH')
        if sample_sequence_length is None:
            sample_sequence_length = total_batch_size//sample_batch_size

        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=config['BUFFER_SIZE']//config['NUM_ENVS'],
            min_length_time_axis=sample_sequence_length,
            add_batch_size=config['NUM_ENVS'],
            sample_batch_size=config['BUFFER_BATCH_SIZE'],
            sample_sequence_length=sample_sequence_length,
            period=period,
        )

        buffer = buffer.replace(
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=0),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample),
        )

        # ---------------
        # use init_transition from 0th env to initialize buffer
        # ---------------
        dummy_rng = jax.random.PRNGKey(0)
        init_preds, action, _ = actor.train_step(
            train_state, init_agent_state, init_timestep, dummy_rng)
        init_transition = vbb.Transition(
            init_timestep,
            action=action,
            extras=FrozenDict(preds=init_preds, agent_state=init_agent_state))
        init_transition_example = jax.tree_map(
            lambda x: x[0], init_transition)

        # [num_envs, max_length, ...]
        buffer_state = buffer.init(init_transition_example)


        # ---------------
        # make buffer for replay. will use same buffer state.
        # ---------------
        total_batch_size = config.get("TOTAL_EXTRA_BATCH_SIZE") or total_batch_size
        sample_batch_size = config['EXTRA_BATCH_SIZE']
        sample_sequence_length = total_batch_size//sample_batch_size
        extra_buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=config['BUFFER_SIZE']//config['NUM_ENVS'],
            min_length_time_axis=sample_sequence_length,
            add_batch_size=config['NUM_ENVS'],
            sample_batch_size=sample_batch_size,
            sample_sequence_length=sample_sequence_length,
            period=period,
        )
        extra_buffer = extra_buffer.replace(
            init=jax.jit(extra_buffer.init),
            add=jax.jit(extra_buffer.add, donate_argnums=0),
            sample=jax.jit(extra_buffer.sample),
            can_sample=jax.jit(extra_buffer.can_sample),
        )
        ##############################
        # INIT Observers
        ##############################
        observer = ObserverCls(
            num_envs=config['NUM_ENVS'],
            log_period=config.get("OBSERVER_PERIOD", 5_000),
            max_num_episodes=config.get("OBSERVER_EPISODES", 200),
        )
        eval_observer = observer

        init_actor_observer_state = observer.init(
            example_timestep=init_timestep,
            example_action=action,
            example_predictions=init_preds)

        init_eval_observer_state = eval_observer.init(
            example_timestep=init_timestep,
            example_action=action,
            example_predictions=init_preds)

        actor_observer_state = observer.observe_first(
            first_timestep=init_timestep,
            observer_state=init_actor_observer_state)

        ##############################
        # INIT LOSS FN
        ##############################
        loss_fn_class = make_loss_fn_class(config)
        loss_fn = loss_fn_class(
            network=agent, logger=logger)

        replay_loss_fn_class = make_replay_loss_fn_class(config)
        replay_loss_fn = replay_loss_fn_class(
            network=agent, logger=logger)

        dummy_rng = jax.random.PRNGKey(0)

        _, dummy_metrics, dummy_grads = vbb.learn_step(
            train_state=train_state,
            rng=dummy_rng,
            buffer=buffer,
            buffer_state=buffer_state,
            loss_fn=loss_fn)

        ##############################
        # DEFINE TRAINING LOOP
        ##############################
        print("="*50)
        print("TRAINING")
        print("="*50)

        def _train_step(old_runner_state: vbb.RunnerState, unused):
            del unused

            ##############################
            # 1. unroll for K steps + add to buffer
            ##############################
            runner_state, traj_batch = vbb.collect_trajectory(
                runner_state=old_runner_state,
                num_steps=config["TRAINING_INTERVAL"],
                actor_step_fn=actor.train_step,
                env_step_fn=vmap_step,
                env_params=train_env_params)

            # things that will be used/changed
            rng = runner_state.rng
            train_state = runner_state.train_state
            buffer_state = runner_state.buffer_state
            # shared_metrics = runner_state.shared_metrics

            # update timesteps count
            timesteps = train_state.timesteps + \
                config["NUM_ENVS"]*config["TRAINING_INTERVAL"]
            # shared_metrics['num_actor_steps'] = timesteps

            train_state = train_state.replace(timesteps=timesteps)

            num_steps, num_envs = traj_batch.timestep.reward.shape
            assert num_steps == config["TRAINING_INTERVAL"]
            assert num_envs == config["NUM_ENVS"]
            # [num_steps, num_envs, ...] -> [num_envs, num_steps, ...]
            buffer_traj_batch = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(x, 0, 1),
                traj_batch
            )

            # update buffer with data of size
            buffer_state = buffer.add(buffer_state, buffer_traj_batch)
            ##############################
            # 2. Learner update
            ##############################
            is_learn_time = (
                (buffer.can_sample(buffer_state))
                & (  # enough experience in buffer
                    timesteps >= config["LEARNING_STARTS"]
                ))

            rng, _rng = jax.random.split(rng)
            train_state, learner_metrics, grads = jax.lax.cond(
                is_learn_time,
                lambda train_state_, rng_: vbb.learn_step(
                    train_state=train_state_,
                    rng=rng_,
                    buffer=buffer,
                    buffer_state=buffer_state,
                    loss_fn=loss_fn),
                lambda train_state, rng: (
                    train_state, dummy_metrics, dummy_grads),  # do nothing
                train_state,
                _rng,
            )

            # update target network
            train_state = jax.lax.cond(
                train_state.n_updates % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda train_state: train_state.replace(
                    target_network_params=jax.tree_map(
                        lambda x: jnp.copy(x), train_state.params)
                ),
                lambda train_state: train_state,
                operand=train_state,
            )

            ##############################
            # 3. Logging learner metrics + evaluation episodes
            ##############################
            # ------------------------
            # log performance information
            # ------------------------
            log_period = max(1, int(config["LEARNER_LOG_PERIOD"]))
            is_log_time = jnp.logical_and(
                is_learn_time, train_state.n_updates % log_period == 0
            )

            train_state = jax.lax.cond(
                is_log_time,
                lambda: train_state.replace(n_logs=train_state.n_logs + 1),
                lambda: train_state,
            )

            jax.lax.cond(
                is_log_time,
                lambda: vbb.log_performance(
                    config=config,
                    agent_reset_fn=agent_reset_fn,
                    actor_train_step_fn=actor.train_step,
                    actor_eval_step_fn=actor.eval_step,
                    env_reset_fn=vmap_reset,
                    env_step_fn=vmap_step,
                    train_env_params=train_env_params,
                    test_env_params=test_env_params,
                    runner_state=runner_state,
                    observer=eval_observer,
                    observer_state=init_eval_observer_state,
                    logger=logger,
                ),
                lambda: None,
            )

            # ------------------------
            # log learner information
            # ------------------------
            loss_name = loss_fn.__class__.__name__
            jax.lax.cond(
                is_log_time,
                lambda: logger.learner_logger(
                    runner_state.train_state, learner_metrics, key=loss_name),
                lambda: None,
            )

            # ------------------------
            # log gradient information
            # ------------------------
            gradient_log_period = config.get("GRADIENT_LOG_PERIOD", 500)
            if gradient_log_period:
                is_log_time = jnp.logical_and(
                    is_learn_time, train_state.n_updates % log_period == 0)
                jax.lax.cond(
                    is_log_time,
                    lambda: logger.gradient_logger(train_state, grads),
                    lambda: None,
                )

            ##############################
            # 4. Creat next runner state
            ##############################
            next_runner_state = runner_state._replace(
                train_state=train_state,
                buffer_state=buffer_state,
                rng=rng)

            return next_runner_state, {}

        ##############################
        # TRAINING LOOP DEFINED. NOW RUN
        ##############################
        # run loop
        rng, _rng = jax.random.split(rng)
        runner_state = vbb.RunnerState(
            train_state=train_state,
            observer_state=actor_observer_state,
            buffer_state=buffer_state,
            timestep=init_timestep,
            agent_state=init_agent_state,
            rng=_rng)

        runner_state, _ = jax.lax.scan(
            _train_step, runner_state, None, config["NUM_UPDATES"]
        )

        #------------------
        # save params
        #------------------
        n_updates = runner_state.train_state.n_updates
        save_params_fn(runner_state.train_state.params, n_updates)

        ##############################
        # DEFINE EXTRA REPLAY LOOP
        ##############################
        print("="*50)
        print("EXTRA REPLAY")
        print("="*50)
        n_updates_pre_replay = runner_state.train_state.n_updates

        def _extra_replay_step(runner_state: vbb.RunnerState, unused):
            del unused

            ##############################
            # 1. Learner update
            ##############################
            # things that will be used/changed
            rng = runner_state.rng
            train_state = runner_state.train_state

            rng, rng_ = jax.random.split(rng)
            train_state, learner_metrics, grads = vbb.learn_step(
                train_state=train_state,
                rng=rng_,
                buffer=extra_buffer,
                buffer_state=runner_state.buffer_state,
                loss_fn=replay_loss_fn)

            # update target network
            train_state = jax.lax.cond(
                train_state.n_updates % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda train_state: train_state.replace(
                    target_network_params=jax.tree_map(
                        lambda x: jnp.copy(x), train_state.params)
                ),
                lambda train_state: train_state,
                operand=train_state,
            )

            ##############################
            # 2. Logging learner metrics
            ##############################
            # ------------------------
            # log learner information
            # ------------------------
            log_period = max(1, int(config["LEARNER_LOG_PERIOD"]))
            is_log_time = train_state.n_updates % log_period == 0
            train_state = jax.lax.cond(
                is_log_time,
                lambda: train_state.replace(n_logs=train_state.n_logs + 1),
                lambda: train_state,
            )

            loss_name = loss_fn.__class__.__name__
            jax.lax.cond(
                is_log_time,
                lambda: logger.learner_logger(
                    runner_state.train_state, learner_metrics, key=loss_name),
                lambda: None,
            )

            # ------------------------
            # log gradient information
            # ------------------------
            gradient_log_period = config.get("GRADIENT_LOG_PERIOD", 500)
            if gradient_log_period:
                is_log_time = train_state.n_updates % log_period == 0
                jax.lax.cond(
                    is_log_time,
                    lambda: logger.gradient_logger(train_state, grads),
                    lambda: None,
                )

            # log performance information
            # ------------------------
            log_period = max(50, int(config["LEARNER_LOG_PERIOD"]))
            is_log_time = train_state.n_updates % log_period == 0
            jax.lax.cond(
                is_log_time,
                lambda: vbb.log_performance(
                    config=config,
                    agent_reset_fn=agent_reset_fn,
                    actor_train_step_fn=actor.train_step,
                    actor_eval_step_fn=actor.eval_step,
                    env_reset_fn=vmap_reset,
                    env_step_fn=vmap_step,
                    train_env_params=train_env_params,
                    test_env_params=test_env_params,
                    runner_state=runner_state,
                    observer=eval_observer,
                    observer_state=init_eval_observer_state,
                    logger=logger,
                ),
                lambda: None,
            )
            ##############################
            # 3. Creat next runner state
            ##############################
            runner_state = runner_state._replace(
                train_state=train_state,
                rng=rng)

            #-----------------
            # optionally save params
            #-----------------
            num_to_save = config.get("NUM_EXTRA_SAVE", 10)
            if num_to_save > 0:
                stride = config["NUM_EXTRA_REPLAY"] // num_to_save
                save_timepoints = jnp.arange(
                    0,
                    config["NUM_EXTRA_REPLAY"],
                    max(stride, 1)) + n_updates_pre_replay

                save_timepoints = save_timepoints[1:]
                should_save = (train_state.n_updates == save_timepoints).any()

                jax.lax.cond(
                    should_save,
                    lambda: save_params_fn(runner_state.train_state.params, train_state.n_updates),
                    lambda: None)

            return runner_state, {}

        ##############################
        # TRAINING LOOP DEFINED. NOW RUN
        ##############################
        # run loop
        runner_state, _ = jax.lax.scan(
            _extra_replay_step, runner_state, None, config["NUM_EXTRA_REPLAY"]
        )
        save_params_fn(runner_state.train_state.params, runner_state.train_state.n_updates)


        return {"runner_state": runner_state}

    return train
