
from typing import Optional, Union, List, Dict

from absl import flags
from absl import logging

import hydra
import os
import datetime
import pickle 

from pathlib import Path
import subprocess

from pprint import pprint
import library.flags

FLAGS = flags.FLAGS


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
  base_path = os.path.join(root_path, folder, search_name)
  for config in configurations:
    if 'group' in config:
      group = config.pop('group')
    else:
      group = search_name

    algo_config, env_config = get_agent_env_configs(
        config=config)

    # dir will be root_path/folder/group/exp_name
    # exp_name is also name in wandb
    trainer_filename_dir = os.path.splitext(os.path.basename(trainer_filename))[0]
    log_dir, exp_name = gen_log_dir(
      base_dir=os.path.join(base_path, trainer_filename_dir, group),
      return_kwpath=True,
      path_skip=['num_steps', 'num_learner_steps', 'group', 'config_name'],
      **algo_config,
      **env_config,
      )

    save_config = dict(
      algo_config=algo_config,
      env_config=env_config,
      wandb_group=group,
      wandb_name=exp_name,
      folder=log_dir,
    )
    save_configs.append(save_config)

  #################################
  # save configs for all runs
  #################################
  base_path = os.path.join(base_path, f'runs-{date_time(True)}')
  process_path(base_path)

  # base_filename = os.path.join(base_path, date_time(time=True))
  configs_file = f"{base_path}/config.pkl"
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

  run_file = f"{base_path}/run.sh"

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
  sbatch_contents += f"#SBATCH -o {base_path}/id=%j.out\n"
  sbatch_contents += f"#SBATCH -e {base_path}/id=%j.err\n"

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
    config_path: str):
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
  elif FLAGS.parallel == 'none':
    if FLAGS.subprocess:  # called by SLURM

      def pickle_load(filename):
        with open(filename, 'rb') as fp:
          config = pickle.load(fp)
          logging.info(f'Loaded: {filename}')
          return config

      configs = pickle_load(FLAGS.search_config)
      config = configs[FLAGS.config_idx-1]  # indexing starts at 1 with SLURM
      run_fn(
        sweep_config=config,
        config_path=config_path,
        save_path=config.get('folder', None),  # load from search config
        )
    else:  # called by this script (i.e. you)
      trainer_filename_dir = os.path.splitext(os.path.basename(trainer_filename))[0]
      save_path = gen_log_dir(
          base_dir=os.path.join(
            folder, trainer_filename_dir, FLAGS.search),
          hourminute=True,
          date=True,
      )
      configs = get_all_configurations(spaces=sweep_fn(FLAGS.search))

      process_path(save_path)
      # this is to make sure that the sweep config has the same format as produced by run_sbatch
      sweep_config = configs[0]
      algo_config, env_config = get_agent_env_configs(config=sweep_config)
      sweep_config['algo_config'] = algo_config
      sweep_config['env_config'] = env_config

      # sensible name for wandb run
      algo_name = algo_config['alg']
      sweep_config['wandb_name'] = f'{algo_name}_{(date_time(time=True))}'

      run_fn(
        sweep_config=sweep_config,
        config_path=config_path,
        save_path=save_path)
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
    print("Sweep Config")
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
    print("Final Config")
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
    dir=save_path,
    # reinit=True,
  )

  try:
    if not FLAGS.wandb:
      wandb_init['mode'] = 'disabled'
  except:
    pass

  return config, wandb_init