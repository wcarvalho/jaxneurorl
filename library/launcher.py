"""
python library/sweep_slurm.py \
  --program=projects/humansf/trainer_housemaze.py \
  --config_path=projects/humansf/configs \
  --time '0-02:30:00' \
  --search=dynaq \

"""
from typing import Optional, Sequence

from absl import logging
import copy
import datetime
import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
import os
from omegaconf import OmegaConf
import subprocess
import pickle
from pprint import pprint
from pathlib import Path
import wandb


def date_time(time: bool = False):
  strkey = ''
  strkey += '%Y.%m.%d-'
  if time:
    strkey += '%H.%M'
  return datetime.datetime.now().strftime(strkey)

def gen_log_dir(
        base_dir="results/",
        date=False,
        hourminute=False,
        seed=None,
        return_kwpath=False,
        path_skip=[],
        **kwargs):

  kwpath = ','.join([f'{key[:4]}={value}' for 
                     key, value in kwargs.items() if not key in path_skip])

  if date:
    job_name = date_time(time=hourminute)
    path = Path(base_dir).joinpath(job_name).joinpath(kwpath)
  else:
    path = Path(base_dir).joinpath(kwpath)

  if seed is not None:
    path = path.joinpath(f'seed={seed}')

  if return_kwpath:
    return str(path), kwpath
  else:
    return str(path)

def make_base_path(
    root_path: str,
    trainer_file: str,
    search: str,
):
  trainer_filename = os.path.splitext(os.path.basename(trainer_file))[0]
  return os.path.join(
      root_path,
      trainer_filename,
      search)

def setup_experiment_config(
    base_path: str,
    run_config: dict,
    datetime_name: bool = True,
    group: str = None,
):

  algo_config, env_config = get_agent_env_configs(run_config=run_config)
  log_dir, wandb_name = gen_log_dir(
      base_dir=os.path.join(base_path, 'save_data', group),
      return_kwpath=True,
      path_skip=['num_steps', 'num_learner_steps', 'group', 'config_name'],
      **algo_config,
      **env_config,
  )
  if datetime_name:
    wandb_name = f'{wandb_name}_{(date_time(time=True))}'

  process_path(log_dir)
  experiment_config = dict(
      run_config=run_config,
      wandb_group=group,
      wandb_name=wandb_name,
      log_dir=log_dir,
  )
  return experiment_config

def process_path(path: str, *subpaths: str) -> str:
  """Process the path string.

  This will process the path string by running `os.path.expanduser` to replace
  any initial "~". It will also append a unique string on the end of the path
  and create the directories leading to this path if necessary.

  Args:
    path: string defining the path to process and create.
    *subpaths: potential subpaths to include after uniqification.
  Returns:
    the processed, expanded path string.
  """
  path = os.path.expanduser(path)
  path = os.path.join(path, *subpaths)
  os.makedirs(path, exist_ok=True)
  return path

def compute_total_combinations(sweep_config):
    parameters = sweep_config.get('parameters', {})
    total_combinations = 1
    for key, values in parameters.items():
        if 'values' in values:
            total_combinations *= len(values['values'])
    return total_combinations

def get_agent_env_configs(
      run_config: dict,
      neither: Sequence[str] = ['group', 'label'],
      default_env_kwargs: Optional[dict] = None):
  """
  Separate config into agent and env configs. Example below. Basically if key starts with "env.", it goes into an env_config.
  Example:
  config = {
    seed: 1,
    width: 2,
    env.room_size: 7,
    group: 'wandb_group4'
  }
  algo_config = {seed: 1, width: 2}
  env_config = {room_size: 7}
  """
  algo_config = dict()
  env_config = dict()

  for k, v in run_config.items():
    if 'env.' in k:
      # e.g. "env.room_size"
      env_config[k.replace("env.", "")] = v
    elif default_env_kwargs and k in default_env_kwargs:
      # e.g. "room_size"
      env_config[k] = v
    elif k in neither:
      pass
    else:
      algo_config[k] = v

  return algo_config, env_config

def default_process_configs(
    hydra_config,
    run_config: dict = None,
    debug: bool = False,
):
  all_env_kwargs = dict()
  run_config = run_config or dict()
  config = dict(hydra_config)

  def update_sub(config_: dict, sub: str):
    sub_config = config_.get(sub, {})
    sub_config_env_kwargs = sub_config.pop('ENV_KWARGS', {})
    all_env_kwargs.update(sub_config_env_kwargs)
    config.update(sub_config)

  algo_config, env_config = get_agent_env_configs(run_config=run_config)

  update_sub(hydra_config, 'app')
  update_sub(hydra_config, 'rlenv')
  update_sub(hydra_config, 'alg')
  update_sub(hydra_config, 'user')

  # update hydra config with algo, env config settings from sweep
  all_env_kwargs.update(env_config)
  config.update(algo_config)
  if debug:
    update_sub(hydra_config, 'debug')

  config['rlenv']['ENV_KWARGS'] = all_env_kwargs
  config['ENV_NAME'] = config["rlenv"].get("ENV_NAME", 'env')

  print("="*50)
  print("Running")
  print("="*50)
  pprint(config)
  return config

def start_wandb_sweep(
    hydra_config: dict,
    sweep_config: dict,
    config_path: str,
    trainer_filename: str,
    folder: str,
    debug=False,
    process_configs_fn=default_process_configs,
    ):
  """For each possible configuration of a run, create a config entry. save a list of all config entries. When SBATCH is called, it will use the ${SLURM_ARRAY_TASK_ID} to run a particular one.
  """
  app_config = hydra_config['app']
  ########
  # prep paths path
  ########
  root_path = str(Path().absolute())
  base_path = make_base_path(
    root_path=os.path.join(root_path, folder),
    trainer_file=trainer_filename,
    search=app_config['search'])

  sbatch_base_path = os.path.join(base_path, 'sbatch', f'runs-{date_time(True)}')
  process_path(sbatch_base_path)

  ########
  # prep sweep config
  ########
  cmd_overrides = HydraConfig.get().overrides.task
  overrides = set(cmd_overrides)
  sweep_config_overrides = sweep_config.pop('overrides', [])
  overrides.update(sweep_config_overrides)
  overrides = list(overrides)

  group = sweep_config.pop('group', app_config['search'])

  if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

  with hydra.initialize(
          version_base=None,
          config_path=config_path):
    hydra_config = hydra.compose(
        config_name='config',
        overrides=overrides)
    hydra_config = OmegaConf.to_container(hydra_config)

  # load the first configuation and use that as the experiment settings
  final_config = process_configs_fn(
    hydra_config,
    run_config=None,  # will load actual configs in subprocesss
    debug=debug)
  assert 'PROJECT' in final_config, 'some config must specify wandb project'

  sweep_config['program'] = trainer_filename
  sweep_config['method'] = app_config['search_method']
  settings_config_file = f"{sbatch_base_path}/config.pkl"
  sweep_config['command'] = [
      'python', trainer_filename,
      f"app.settings_config={settings_config_file}",
       "app.subprocess=True",
       f"app.PROJECT={final_config['PROJECT']}",
       f"app.base_path={base_path}",
       f"app.group={group}"
       ] + sweep_config_overrides
  if debug:
    sweep_config['command'].append(
      'app.debug=True'
    )
  print("="*50)
  print("Sweep config")
  print("="*50)
  pprint(sweep_config)
  sweep_id = wandb.sweep(
      sweep_config,
      project=final_config['PROJECT'],
      entity=final_config["entity"],
      )

  with open(settings_config_file, 'wb') as fp:
      pickle.dump(dict(sweep_id=sweep_id), fp)
      logging.info(f'Saved: {settings_config_file}')

  #################################
  # create run.sh file to run with sbatch
  #################################
  python_file_command = " ".join(sweep_config['command'])
  run_file = f"{sbatch_base_path}/run.sh"

  if debug:
    # create file and run single python command
    run_file_contents = "#!/bin/bash\n" + python_file_command
    print("-"*20)
    print(run_file_contents)
    print("-"*20)
    with open(run_file, 'w') as file:
      # Write the string to the file
      file.write(run_file_contents)
    process = subprocess.Popen(f"chmod +x {run_file}", shell=True)
    process.wait()
    process = subprocess.Popen(run_file, shell=True)
    process.wait()
    return

  #################################
  # create sbatch file
  #################################
  hourminute =  datetime.datetime.now().strftime("%H:%M")
  year =  datetime.datetime.now().strftime("%Y")
  job_name = f"{hourminute}-{app_config['search']}-{year}"
  process_path(sbatch_base_path)

  sbatch_contents = "#!/bin/bash\n"
  sbatch_contents += f"#SBATCH --gres=gpu:{app_config['num_gpus']}\n"
  sbatch_contents += f"#SBATCH -c {app_config['num_cpus']}\n"
  sbatch_contents += f"#SBATCH --mem {app_config['memory']}\n"
  sbatch_contents += f"#SBATCH -t {app_config['time']}\n"
  sbatch_contents += f"#SBATCH -J {job_name}\n"

  sbatch_contents += f"#SBATCH -p {app_config['partition']}\n"
  sbatch_contents += f"#SBATCH --account {app_config['account']}\n"
  sbatch_contents += f"#SBATCH -o {sbatch_base_path}/id=%j.out\n"
  sbatch_contents += f"#SBATCH -e {sbatch_base_path}/id=%j.err\n"

  wandb_api_key = final_config['wandb_api_key']
  sbatch_contents += f"export WANDB_API_KEY={wandb_api_key}\n"
  sbatch_contents += sbatch_contents + python_file_command

  #project = final_config['PROJECT']
  #entity = final_config["entity"]
  #sbatch_contents += f"echo 'logging into wandb...'\n"
  #sbatch_contents += f"wandb login {wandb_api_key} && "
  #sbatch_contents += f"echo 'Checking wandb service status...' && "
  #sbatch_contents += f"wandb status && "
  #sbatch_contents += f"echo 'Running wandb agent...' && "
  #sbatch_contents += f"wandb agent --count 1 {entity}/{project}/{sweep_id}\n"
  print("-"*20)
  print(sbatch_contents)
  print("-"*20)
  run_file = f"{sbatch_base_path}/run.sh"
  with open(run_file, 'w') as file:
    # Write the string to the file
    file.write(sbatch_contents)

  total_jobs = compute_total_combinations(sweep_config)
  max_concurrent = app_config['max_concurrent']
  sbatch_command = f"sbatch --array=1-{total_jobs}%{max_concurrent} {run_file}"
  pprint(sbatch_command)
  process = subprocess.Popen(f"chmod +x {run_file}", shell=True)
  process.wait()
  process = subprocess.Popen(sbatch_command, shell=True)
  process.wait()

def start_vanilla_sweep(
    hydra_config: dict,
    sweep_config: dict,
    config_path: str,
    trainer_filename: str,
    folder: str,
    debug=False,
    process_configs_fn=default_process_configs,
    ):
  """For each possible configuration of a run, create a config entry. save a list of all config entries. When SBATCH is called, it will use the ${SLURM_ARRAY_TASK_ID} to run a particular one.
  """
  app_config = hydra_config['app']
  ########
  # prep paths path
  ########
  root_path = str(Path().absolute())
  base_path = make_base_path(
    root_path=os.path.join(root_path, folder),
    trainer_file=trainer_filename,
    search=app_config['search'])

  sbatch_base_path = os.path.join(base_path, 'sbatch', f'runs-{date_time(True)}')
  process_path(sbatch_base_path)

  #################################
  # create configs for all runs
  #################################
  group = sweep_config.pop('group', app_config['search'])
  configurations = get_all_configurations(sweep_config['parameters'])
  logging.info("searching:")
  pprint(configurations)

  experiment_configs = []
  base_path = make_base_path(
    root_path=os.path.join(root_path, folder),
    trainer_file=trainer_filename,
    search=app_config['search'])

  for config in configurations:
    experiment_config = setup_experiment_config(
      base_path=base_path,
      group=group,
      run_config=config,
      datetime_name=False,
    )
    experiment_configs.append(experiment_config)

  #################################
  # save configs for all runs
  #################################
  settings_config_file = f"{sbatch_base_path}/config.pkl"
  with open(settings_config_file, 'wb') as fp:
      pickle.dump(experiment_configs, fp)
      logging.info(f'Saved: {settings_config_file}')

  ########
  # prep sweep file command
  ########
  cmd_overrides = HydraConfig.get().overrides.task
  overrides = set(cmd_overrides)
  sweep_config_overrides = sweep_config.pop('overrides', [])
  overrides.update(sweep_config_overrides)
  overrides = list(overrides)

  group = sweep_config.pop('group', app_config['search'])

  if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

  with hydra.initialize(
          version_base=None,
          config_path=config_path):
    hydra_config = hydra.compose(
        config_name='config',
        overrides=overrides)
    hydra_config = OmegaConf.to_container(hydra_config)

  # load the first configuation and use that as the experiment settings
  final_config = process_configs_fn(
    hydra_config,
    run_config=None,  # will load actual configs in subprocesss
    debug=debug)
  assert 'PROJECT' in final_config, 'some config must specify wandb project'

  python_file_contents = [
      'python', trainer_filename,
      f"app.settings_config={settings_config_file}",
       "app.subprocess=True",
       f"app.PROJECT={final_config['PROJECT']}",
       f"app.base_path={base_path}",
       f"app.group={group}"
       ] + sweep_config_overrides

  if debug:
    python_file_contents += [f"app.config_idx=1", 'app.debug=True']
  else:
    python_file_contents += [f"app.config_idx=$SLURM_ARRAY_TASK_ID"]

  #################################
  # create run.sh file to run with sbatch
  #################################
  python_file_command = " ".join(python_file_contents)
  run_file = f"{sbatch_base_path}/run.sh"

  if debug:
    # create file and run single python command
    run_file_contents = "#!/bin/bash\n" + python_file_command
    print("-"*20)
    print(run_file_contents)
    print("-"*20)
    with open(run_file, 'w') as file:
      # Write the string to the file
      file.write(run_file_contents)
    process = subprocess.Popen(f"chmod +x {run_file}", shell=True)
    process.wait()
    process = subprocess.Popen(run_file, shell=True)
    process.wait()
    return

  #################################
  # create sbatch file
  #################################
  hourminute =  datetime.datetime.now().strftime("%H:%M")
  year =  datetime.datetime.now().strftime("%Y")
  job_name = f"{hourminute}-{app_config['search']}-{year}"
  process_path(sbatch_base_path)

  sbatch_contents = "#!/bin/bash\n"
  sbatch_contents += f"#SBATCH --gres=gpu:{app_config['num_gpus']}\n"
  sbatch_contents += f"#SBATCH -c {app_config['num_cpus']}\n"
  sbatch_contents += f"#SBATCH --mem {app_config['memory']}\n"
  sbatch_contents += f"#SBATCH -t {app_config['time']}\n"
  sbatch_contents += f"#SBATCH -J {job_name}\n"

  sbatch_contents += f"#SBATCH -p {app_config['partition']}\n"
  sbatch_contents += f"#SBATCH --account {app_config['account']}\n"
  sbatch_contents += f"#SBATCH -o {sbatch_base_path}/id=%j.out\n"
  sbatch_contents += f"#SBATCH -e {sbatch_base_path}/id=%j.err\n"
  sbatch_contents += sbatch_contents + python_file_command

  print("-"*20)
  print(sbatch_contents)
  print("-"*20)
  run_file = f"{sbatch_base_path}/run.sh"
  with open(run_file, 'w') as file:
    # Write the string to the file
    file.write(sbatch_contents)

  total_jobs = compute_total_combinations(sweep_config)
  max_concurrent = app_config['max_concurrent']
  sbatch_command = f"sbatch --array=1-{total_jobs}%{max_concurrent} {run_file}"
  pprint(sbatch_command)
  process = subprocess.Popen(f"chmod +x {run_file}", shell=True)
  process.wait()
  process = subprocess.Popen(sbatch_command, shell=True)
  process.wait()

def run_wandb_sweep_run(
    run_fn,
    hydra_config: dict,
    folder: str,
    process_configs_fn=default_process_configs,
  ):

  def wrapped_run_fn():
    wandb_init = dict(
      project=hydra_config['app']['PROJECT'],
      entity=hydra_config['user']['entity'],
      group=hydra_config['app']['group'],
      save_code=True,
      mode='offline',
      dir=folder
      )
    if not hydra_config['app']['wandb']:
      wandb_init['mode'] = 'disabled'

    wandb.init(**wandb_init)

    run_config = wandb.config.as_dict()
    default_config = OmegaConf.to_container(hydra_config)
    final_config = process_configs_fn(
      default_config,
      run_config,
      debug=hydra_config['app']['debug'])

    experiment_config = setup_experiment_config(
        base_path=hydra_config['app']['base_path'],
        run_config=run_config,
        group=hydra_config['app']['group'],
        datetime_name=False,
    )

    # Update wandb configuration
    wandb_kwargs = dict(
      #group=final_config['group'],
      #entity=final_config["entity"],
      name=experiment_config['wandb_name'],
      #dir=experiment_config['log_dir'],
    )
    for key, value in wandb_kwargs.items():
      setattr(wandb.run, key, value)

    run_fn(
        config=final_config,
        save_path=experiment_config['log_dir']
    )

  filename = hydra_config['app']['settings_config']
  with open(filename, 'rb') as fp:
    settings = pickle.load(fp)
    sweep_id = settings['sweep_id']
    logging.info(f'Loaded sweep_id: {sweep_id}')

  wandb.agent(sweep_id, wrapped_run_fn, count=1)

def run_vanilla_sweep_run(
    run_fn,
    hydra_config: dict,
    folder: str,
    process_configs_fn=default_process_configs,
  ):
  del folder
  filename = hydra_config['app']['settings_config']
  with open(filename, 'rb') as fp:
    settings = pickle.load(fp)

  experiment_config = settings[hydra_config['app']['config_idx']-1]
  default_config = OmegaConf.to_container(hydra_config)
  final_config = process_configs_fn(
    default_config,
    experiment_config['run_config'],
    debug=hydra_config['app']['debug'])

  wandb_init = dict(
    project=hydra_config['app']['PROJECT'],
    entity=hydra_config['user']['entity'],
    group=experiment_config['wandb_group'],
    name=experiment_config['wandb_name'],
    dir=experiment_config['log_dir'],
    save_code=True,
    config=final_config,
    )
  if not hydra_config['app']['wandb']:
    wandb_init['mode'] = 'disabled'

  wandb.init(**wandb_init)

  run_fn(
      config=final_config,
      save_path=experiment_config['log_dir']
  )


def run_individual(
    run_fn,
    app_config: dict,
    sweep_config: dict,
    config_path: str,
    trainer_filename: str,
    folder: str,
    process_configs_fn=default_process_configs,
    debug: bool = False,
  ):
  ########
  # make path
  ########
  base_path = make_base_path(
      root_path=f"{folder}_single",
      trainer_file=trainer_filename,
      search=app_config['search'])

  ########
  # prep sweep config
  ########
  # this is to make sure that the sweep config has the same format as produced by run_sbatch
  cmd_overrides = HydraConfig.get().overrides.task
  overrides = set(cmd_overrides)
  overrides.update(sweep_config.pop('overrides', []))
  overrides = list(overrides)
  group = sweep_config.pop('group', app_config['search'])

  if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

  with hydra.initialize(
          version_base=None,
          config_path=config_path):
    hydra_config = hydra.compose(
        config_name='config',
        overrides=overrides)
    hydra_config = OmegaConf.to_container(hydra_config)

  # load the first configuation and use that as the experiment settings
  configs = get_all_configurations(sweep_config['parameters'])
  run_config = configs[0]
  final_config = process_configs_fn(
    hydra_config=hydra_config,
    run_config=run_config,
    debug=debug)

  experiment_config = setup_experiment_config(
      base_path=base_path,
      run_config=run_config,
      group=group,
      datetime_name=True,
  )
  
  # -------------------
  # load wandb args
  # -------------------
  algo_name = final_config.get('alg', None)
  env_name = final_config["rlenv"].get("ENV_NAME", 'env')
  project = final_config["PROJECT"]
  if app_config['debug']:
    project += "_debug"

  wandb_init = dict(
      project=project,
      entity=final_config["entity"],
      group=group,
      name=experiment_config.get('wandb_name', f'{algo_name}_{env_name}'),
      config=final_config,
      save_code=False,
      dir=experiment_config['log_dir'],
  )
  if not app_config['wandb']:
    wandb_init['mode'] = 'disabled'

  wandb.init(**wandb_init)
  run_fn(
    config=final_config,
    save_path=experiment_config['log_dir']
  )

def get_all_configurations(parameters_config: dict):
  import itertools
  parameters_config = {k: v['values'] for k,v in parameters_config.items()}

  keys, value_lists = zip(*[(k, v) for k, v in parameters_config.items()])

  # Generate the Cartesian product of the value lists
  cartesian_product = itertools.product(*value_lists)

  # Create a list of dictionaries for each combination
  configs = [dict(zip(keys, values)) for values in cartesian_product]
  return configs

def run(
    hydra_config,
    run_fn,
    sweep_fn,
    folder: str,
    trainer_filename: str,
    config_path: str,
    process_configs_fn=default_process_configs,
  ):
  config_path = os.path.join("..", config_path)
  if hydra_config['app']['parallel'] == 'sbatch':
    if hydra_config['app']['wandb_search']:
      return start_wandb_sweep(
        hydra_config=hydra_config,
        sweep_config=sweep_fn(hydra_config['app']['search']),
        config_path=config_path,
        trainer_filename=trainer_filename,
        process_configs_fn=process_configs_fn,
        folder=folder,
        debug=hydra_config['app']['debug_sweep'],
      )
    else:
      return start_vanilla_sweep(
        hydra_config=hydra_config,
        sweep_config=sweep_fn(hydra_config['app']['search']),
        config_path=config_path,
        trainer_filename=trainer_filename,
        process_configs_fn=process_configs_fn,
        folder=folder,
        debug=hydra_config['app']['debug_sweep'],
      )

  elif hydra_config['app']['parallel'] == 'none':
    # -------------------
    # load experiment config. varies based on whether called by slurm or not.
    # -------------------
    if hydra_config['app']['subprocess']:  # called by SLURM
      if hydra_config['app']['wandb_search']:
        return run_wandb_sweep_run(
          run_fn=run_fn,
          hydra_config=hydra_config,
          process_configs_fn=process_configs_fn,
          folder=folder,
        )
      else:
        return run_vanilla_sweep_run(
          run_fn=run_fn,
          hydra_config=hydra_config,
          process_configs_fn=process_configs_fn,
          folder=folder,
        )

    else:  # called by this script (i.e. you)
      # mimics the structure of sweep so file naming is consistent
      return run_individual(
        run_fn=run_fn,
        app_config=hydra_config['app'],
        sweep_config=sweep_fn(hydra_config['app']['search']),
        config_path=config_path,
        trainer_filename=trainer_filename,
        folder=folder,
        process_configs_fn=process_configs_fn,
        debug=hydra_config['app']['debug'],
      )

  else:
    raise NotImplementedError