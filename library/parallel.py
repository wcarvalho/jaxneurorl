
from typing import Optional, Union, List, Dict

from absl import flags
from absl import logging

import hydra
import os
import datetime
import pickle
import jax
import wandb

from pathlib import Path
import subprocess

from pprint import pprint
import library.flags

FLAGS = flags.FLAGS

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
    group: str,
    config: dict,
    datetime_name: bool = True,
    ):

  algo_config, env_config = get_agent_env_configs(config=config)
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
  config = dict(
      algo_config=algo_config,
      env_config=env_config,
      wandb_group=group,
      wandb_name=wandb_name,
      log_dir=log_dir,
    )
  return config

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

def date_time(time: bool=False):
  strkey = '%Y.%m.%d'
  if time:
    strkey += '-%H.%M'
  return datetime.datetime.now().strftime(strkey)

def gen_log_dir(
    base_dir="results/",
    date=False,
    hourminute=False,
    seed=None,
    return_kwpath=False,
    path_skip=[],
    **kwargs):

  kwpath = ','.join([f'{key[:4]}={value}' for key, value in kwargs.items() if not key in path_skip])

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

def get_all_configurations(spaces: Union[Dict, List[Dict]]):
    import itertools
    all_settings = []
    if isinstance(spaces, dict):
      spaces = [spaces]
    for space in spaces:
      # Extract keys and their corresponding lists from the space dictionary
      keys, value_lists = zip(*[(key, space[key]['grid_search']) for key in space])

      # Generate the Cartesian product of the value lists
      cartesian_product = itertools.product(*value_lists)

      # Create a list of dictionaries for each combination
      all_settings += [dict(zip(keys, values)) for values in cartesian_product]

    return all_settings

def get_agent_env_configs(
    config: dict,
    neither: List[str] = ['group', 'label'],
    default_env_kwargs: Optional[dict]=None):
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

  for k, v in config.items():
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

def run_sbatch(
    trainer_filename: str,
    folder: str,
    search_name: str,
    spaces: Union[Dict, List[Dict]],
    debug: bool = False):
  """For each possible configuration of a run, create a config entry. save a list of all config entries. When SBATCH is called, it will use the ${SLURM_ARRAY_TASK_ID} to run a particular one.
  """

  #################################
  # create configs for all runs
  #################################
  root_path = str(Path().absolute())
  configurations = get_all_configurations(spaces=spaces)

  from pprint import pprint
  logging.info("searching:")
  pprint(configurations)

  save_configs = []
  base_path = make_base_path(
    root_path=os.path.join(root_path, folder),
    trainer_file=trainer_filename,
    search=search_name)

  for config in configurations:

    group = config.pop('group', search_name)
    save_config = setup_experiment_config(
      base_path=base_path,
      group=group,
      config=config,
      datetime_name=False,
    )

    save_configs.append(save_config)

  #################################
  # save configs for all runs
  #################################
  sbatch_base_path = os.path.join(base_path, 'sbatch', f'runs-{date_time(True)}')
  process_path(sbatch_base_path)

  # base_filename = os.path.join(sbatch_base_path, date_time(time=True))
  configs_file = f"{sbatch_base_path}/config.pkl"
  with open(configs_file, 'wb') as fp:
      pickle.dump(save_configs, fp)
      logging.info(f'Saved: {configs_file}')

  #################################
  # create run.sh file to run with sbatch
  #################################
  python_file_contents = f"python {trainer_filename}"
  python_file_contents += f" --search_config={configs_file}"
  # python_file_contents += f" --use_wandb={use_wandb}"
  if debug:
    python_file_contents += f" --config_idx=1"
  else:
    python_file_contents += f" --config_idx=$SLURM_ARRAY_TASK_ID"
  # python_file_contents += f" --run_distributed={run_distributed}"
  python_file_contents += f" --subprocess={True}"
  # python_file_contents += f" --make_path={False}"

  run_file = f"{sbatch_base_path}/run.sh"

  if debug:
    # create file and run single python command
    run_file_contents = "#!/bin/bash\n" + python_file_contents
    logging.warning("only running first config")
    print(run_file_contents)
    with open(run_file, 'w') as file:
      # Write the string to the file
      file.write(run_file_contents)
    process = subprocess.Popen(f"chmod +x {run_file}", shell=True)
    process = subprocess.Popen(run_file, shell=True)
    process.wait()
    return

  #################################
  # create sbatch file
  #################################
  job_name=f'{search_name}-{date_time(True)}'
  sbatch_contents = f"#SBATCH --gres=gpu:{FLAGS.num_gpus}\n"
  sbatch_contents += f"#SBATCH -c {FLAGS.num_cpus}\n"
  sbatch_contents += f"#SBATCH --mem {FLAGS.memory}\n"
  sbatch_contents += f"#SBATCH -J {job_name}\n"

  # sbatch_contents += f"#SBATCH --mem-per-cpu={FLAGS.memory}\n"
  sbatch_contents += f"#SBATCH -p {FLAGS.partition}\n"
  sbatch_contents += f"#SBATCH -t {FLAGS.time}"
  sbatch_contents += f"#SBATCH --account {FLAGS.account}\n"
  sbatch_contents += f"#SBATCH -o {sbatch_base_path}/id=%j.out\n"
  sbatch_contents += f"#SBATCH -e {sbatch_base_path}/id=%j.err\n"

  run_file_contents = "#!/bin/bash\n" + sbatch_contents + python_file_contents
  print("-"*20)
  print(run_file_contents)
  print("-"*20)
  with open(run_file, 'w') as file:
    # Write the string to the file
    file.write(run_file_contents)

  total_jobs = len(save_configs)
  max_concurrent = min(FLAGS.max_concurrent, total_jobs)
  sbatch_command = f"sbatch --array=1-{total_jobs}%{max_concurrent} {run_file}"
  logging.info(sbatch_command)
  process = subprocess.Popen(f"chmod +x {run_file}", shell=True)
  process = subprocess.Popen(sbatch_command, shell=True)
  process.wait()

def run(
    run_fn,
    sweep_fn,
    folder: str,
    trainer_filename: str,
    config_path: str,
    remove_wandb_files: bool = True):
  """The basic logics is as follows:

  Simplest: if FLAGS.parallel == 'none', simply runs run_fn
    - here, we use the first config setting from sweep_fn

  if FLAGS.parallel == 'sbatch':
    - first a set of configs are made using the sweep_fn
    - they are then saved and trainer_filename is called again
    - this function is called again from a sub-process (1 for each SLURM launch)
      - in this subprocess, the corresponding config is taken
      - then run_fn is called again, but it combines the hyda config defaults with the config settings in this sub-process (i.e. this part of the sweep) 

  Args:
      run_fn (-): _description_
      sweep_fn (_type_): _description_
      folder (_type_): _description_
      trainer_filename (str): _description_
      config_path (str): _description_

  Raises:
      NotImplementedError: _description_
  """
  config_path = os.path.join("..", config_path)
  if FLAGS.parallel == 'sbatch':
    run_sbatch(
      trainer_filename=trainer_filename,
      folder=folder,
      search_name=FLAGS.search,
      debug=FLAGS.debug_parallel,
      spaces=sweep_fn(FLAGS.search),
    )
    return
  elif FLAGS.parallel == 'none':
    #-------------------
    # load experiment config. varies based on whether called by slurm or not.
    #-------------------
    if FLAGS.subprocess:  # called by SLURM

      def pickle_load(filename):
        with open(filename, 'rb') as fp:
          config = pickle.load(fp)
          logging.info(f'Loaded: {filename}')
          return config

      configs = pickle_load(FLAGS.search_config)
      experiment_config = configs[FLAGS.config_idx-1]  # indexing starts at 1 with SLURM
    else:  # called by this script (i.e. you)
      configs = get_all_configurations(spaces=sweep_fn(FLAGS.search))

      base_path = make_base_path(
        root_path=f"{folder}_single",
        trainer_file=trainer_filename,
        search=FLAGS.search)

      # this is to make sure that the sweep config has the same format as produced by run_sbatch
      config = configs[0]
      experiment_config = setup_experiment_config(
        base_path=base_path,
        group=config.pop('group', FLAGS.search),
        config=config,
        datetime_name=True,
      )
    #-------------------
    # load hyra config from experiment config
    # run experiment and optionally remove wandb files
    #-------------------
    config, wandb_init = load_hydra_config(
      sweep_config=experiment_config,
      config_path=config_path,
      save_path=experiment_config['log_dir'],
      tags=[f"jax_{jax.__version__}"]
      )
    wandb.init(**wandb_init)
    run_fn(
      config=config,
      save_path=experiment_config['log_dir'])
    if remove_wandb_files:
      #---------------
      # clean up wandb dir
      #---------------
      wandb_dir = wandb_init.get("dir", './wandb')
      if os.path.exists(wandb_dir):
        import shutil
        shutil.rmtree(wandb_dir)

  else:
    raise NotImplementedError

def load_hydra_config(
    sweep_config,
    config_path: str,
    save_path: Optional[str] = None,
    process_sweep_config: bool = False,
    verbose: bool = True,
    tags=[]):
  from omegaconf import OmegaConf

  if process_sweep_config:
    algo_config, env_config = get_agent_env_configs(config=sweep_config)
    sweep_config['algo_config'] = algo_config
    sweep_config['env_config'] = env_config
  #---------------
  # load algorithm, config, and env names
  #---------------
  if verbose:
    print("="*50)
    print("Sweep Config")
    print("="*50)
    pprint(sweep_config)
  #---------------
  # split sweep config into algo + env configs
  #---------------
  sweep_algo_config = sweep_config['algo_config']
  sweep_env_config = sweep_config['env_config']

  algo_name = sweep_algo_config.get('alg', None)
  assert algo_name is not None, "set algorithm"
  config_name = sweep_algo_config.pop('config_name', algo_name)

  #---------------
  # load & update hydra config
  #---------------
  with hydra.initialize(
    version_base=None,
    config_path=config_path):
    config = hydra.compose(
      config_name=config_name)
    config = OmegaConf.to_container(config)

    # some setup to make sure env field is populated
    hydra_env_config = config.get('env', {})
    hydra_env_kwargs = hydra_env_config.pop('ENV_KWARGS', {})

    # put everything in env config in main config
    config.update(hydra_env_config)

    config['env']['ENV_KWARGS'] = hydra_env_kwargs

  # update hydra config with env config settings from sweep
  config['env']['ENV_KWARGS'].update(sweep_env_config)
  config['ENV_NAME'] = env_name = config["env"].get("ENV_NAME", 'env')
 
  # update hydra config with algo config settings from sweep
  config.update(sweep_algo_config)


  try:
    if FLAGS.debug:
      config.update(config.pop('debug', {}))
  except:
    pass

  if verbose:
    print("="*50)
    print("Final Config")
    print("="*50)
    pprint(config)

  #---------------
  # create wandb kwargs
  #---------------
  project = config["PROJECT"]
  try:
    if FLAGS.debug:
      project += "_debug"
  except:
    pass

  wandb_init = dict(
    project=project,
    entity=config['user']["ENTITY"],
    group=sweep_config.get('wandb_group', 'default'),
    name=sweep_config.get('wandb_name', f'{algo_name}_{env_name}'),
    tags=tags+[algo_name.upper(), env_name.upper()],
    config=config,
    save_code=False,
    dir=os.path.join(save_path, 'wandb'),
  )

  try:
    if not FLAGS.wandb:
      wandb_init['mode'] = 'disabled'
  except:
    pass

  return config, wandb_init

