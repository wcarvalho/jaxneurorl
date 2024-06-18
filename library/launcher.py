"""
python library/sweep_slurm.py \
  --program=projects/humansf/trainer_housemaze.py \
  --config_path=projects/humansf/configs \
  --time '0-02:30:00' \
  --search=dynaq \

"""
from typing import Optional, Sequence

from absl import logging
import datetime
import hydra
import importlib
from pprint import pprint
from pathlib import Path
import pickle
import subprocess
import os
import wandb


def pickle_load(filename):
  with open(filename, 'rb') as fp:
    config = pickle.load(fp)
    logging.info(f'Loaded: {filename}')
    return config


def date_time(time: bool = False):
  strkey = ''
  if time:
    strkey = '%H.%M-'
  strkey = '%Y.%m.%d'
  return datetime.datetime.now().strftime(strkey)

def gen_log_dir(
        base_dir="results/",
        date=False,
        hourminute=False,
        seed=None,
        return_kwpath=False,
        path_skip=[],
        **kwargs):

  kwpath = ','.join([f'{key[:4]}={value}' for key,
                    value in kwargs.items() if not key in path_skip])

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

def get_function(module_path, function_name):
  # Function to import a module and get a function from it
  # Convert file path to module path
  module_path = module_path.replace('/', '.').replace('.py', '')

  # Import the module
  module = importlib.import_module(module_path)

  # Get the function from the module
  func = getattr(module, function_name)
  return func

def compute_total_combinations(sweep_config):
    parameters = sweep_config.get('parameters', {})
    total_combinations = 1
    for key, values in parameters.items():
        if 'values' in values:
            total_combinations *= len(values['values'])
    return total_combinations

def get_agent_env_configs(
      config: dict,
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

def load_hydra_config(
    config_path: str,
    experiment_config = None,
    process_experiment_config: bool = False,
    verbose: bool = True,
    debug: bool = False):

  experiment_config = experiment_config or {}
  from omegaconf import OmegaConf
  if process_experiment_config:
    algo_config, env_config = get_agent_env_configs(config=experiment_config)
    experiment_config['algo_config'] = algo_config
    experiment_config['env_config'] = env_config
  # ---------------
  # load algorithm, config, and env names
  # ---------------
  if verbose:
    print("="*50)
    print("Experiment Config")
    print("="*50)
    print(experiment_config)
  # ---------------
  # split sweep config into algo + env configs
  # ---------------
  sweep_algo_config = experiment_config['algo_config']
  sweep_env_config = experiment_config['env_config']

  algo_name = sweep_algo_config.get('alg', None)
  assert algo_name is not None, "set algorithm"
  config_name = sweep_algo_config.pop('config_name')

  # ---------------
  # load & update hydra config
  # ---------------
  all_env_kwargs = dict()

  def update_sub(config: dict, sub: str):
    sub_config = config.get(sub, {})
    sub_config_env_kwargs = sub_config.pop('ENV_KWARGS', {})
    config.update(sub_config)
    all_env_kwargs.update(sub_config_env_kwargs)
    return config

  with hydra.initialize(
          version_base=None,
          config_path=config_path):
    config = hydra.compose(
        config_name=config_name)
    config = OmegaConf.to_container(config)

    # some setup to make sure env field is populated
    config = update_sub(config, sub='rlenv')

  # update hydra config with env config settings from sweep
  all_env_kwargs.update(sweep_env_config)
  config['ENV_NAME'] = env_name = config["rlenv"].get("ENV_NAME", 'env')

  # update hydra config with algo config settings from sweep
  config.update(sweep_algo_config)

  if debug:
    config = update_sub(config, sub='debug')

  config['rlenv']['ENV_KWARGS'] = all_env_kwargs

  if verbose:
    print("="*50)
    print("Final Config")
    print("="*50)
    pprint(config)

  return config


def run_sweep(
    args,
    sweep_config: dict,
    config_path: str,
    trainer_filename: str,
    load_config_fn,
    folder: str,
    debug=False,
    ):
  """For each possible configuration of a run, create a config entry. save a list of all config entries. When SBATCH is called, it will use the ${SLURM_ARRAY_TASK_ID} to run a particular one.
  """
  ########
  # make path
  ########
  import ipdb; ipdb.set_trace()
  root_path = str(Path().absolute())
  base_path = make_base_path(
    root_path=os.path.join(root_path, folder),
    trainer_file=trainer_filename,
    search=args.search)

  sbatch_base_path = os.path.join(
      base_path, 'sbatch', f'runs-{date_time(True)}')
  
  ########
  # prep sweep config
  ########
  default_config = load_config_fn(
      config_path=config_path,
      verbose=False)
  experiment_config = setup_experiment_config(
      base_path=base_path,
      group=default_config.pop('group', args.search),
      config=default_config,
      datetime_name=False,
  )
  settings_config = dict(
    wandb_kwargs=dict(
        project=default_config['PROJECT'],
        entity=default_config["ENTITY"],
        group=experiment_config['wandb_group'],
        name=experiment_config['wandb_name'],
        save_code=False,
        dir=experiment_config['log_dir'],
    )
  )
  settings_config_file = f"{sbatch_base_path}/config.pkl"
  with open(settings_config_file, 'wb') as fp:
      pickle.dump(settings_config, fp)
      logging.info(f'Saved: {settings_config_file}')

  sweep_config['program'] = args.program
  sweep_config['method'] = args.search_method
  sweep_config['command'] = [
      'python', trainer_filename,
      f"--settings_config={settings_config_file}",
       "--subprocess=True"
  ]
  if debug:
    sweep_config['command'].append(
      '--debug=True'
    )

  sweep_id = wandb.sweep(
      sweep_config,
      entity=default_config['PROJECT'],
      project=default_config["ENTITY"])

  if debug:
    # create file and run single python command
    command = f"wandb agent --count 1 {sweep_id}"
    process = subprocess.Popen(command, shell=True)
    process.wait()
    return
  #################################
  # create sbatch file
  #################################

  job_name = f'{date_time(True)}-{args.search}'
  process_path(sbatch_base_path)

  sbatch_contents = f"#SBATCH --gres=gpu:{args.num_gpus}\n"
  sbatch_contents += f"#SBATCH -c {args.num_cpus}\n"
  sbatch_contents += f"#SBATCH --mem {args.memory}\n"
  sbatch_contents += f"#SBATCH -t {args.time}"
  sbatch_contents += f"#SBATCH -J {job_name}\n"

  sbatch_contents += f"#SBATCH -p {args.partition}\n"
  sbatch_contents += f"#SBATCH --account {args.account}\n"
  sbatch_contents += f"#SBATCH -o {sbatch_base_path}/id=%j.out\n"
  sbatch_contents += f"#SBATCH -e {sbatch_base_path}/id=%j.err\n"

  run_file_contents = sbatch_contents + f"wandb agent --count 1 {sweep_id}"
  print("-"*20)
  print(run_file_contents)
  print("-"*20)
  run_file = f"{sbatch_base_path}/run.sh"
  with open(run_file, 'w') as file:
    # Write the string to the file
    file.write(run_file_contents)

  total_jobs = compute_total_combinations(sweep_config)
  max_concurrent = args.max_concurrent
  sbatch_command = f"sbatch --array=1-{total_jobs}%{max_concurrent} {run_file}"
  pprint(sbatch_command)
  process = subprocess.Popen(f"chmod +x {run_file}", shell=True)
  process = subprocess.Popen(sbatch_command, shell=True)
  process.wait()

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
    args,
    run_fn,
    sweep_fn,
    folder: str,
    trainer_filename: str,
    config_path: str,
    load_config_fn=load_hydra_config,
  ):

  config_path = os.path.join("..", config_path)

  if args.parallel == 'sbatch':
    run_sweep(
      args=args,
      sweep_config=sweep_fn(args.search),
      config_path=config_path,
      trainer_filename=trainer_filename,
      load_config_fn=load_config_fn,
      folder=folder,
      debug=args.debug_sweep,
    )
    return

  elif args.parallel == 'none':
    # -------------------
    # load experiment config. varies based on whether called by slurm or not.
    # -------------------
    if args.subprocess:  # called by SLURM
      settings_config = pickle_load(args.settings_config)
      wandb.init(**settings_config['wandb_kwargs'])

      experiment_config = wandb.config.as_dict()
      config = load_config_fn(
          experiment_config=experiment_config,
          config_path=config_path,
      )
      run_fn(
          config=config,
          save_path=experiment_config['log_dir']
      )
      return

    else:  # called by this script (i.e. you)
      base_path = make_base_path(
          root_path=f"{folder}_single",
          trainer_file=trainer_filename,
          search=args.search)

      # this is to make sure that the sweep config has the same format as produced by run_sbatch
      sweep_config = sweep_fn(args.search)
      configs = get_all_configurations(sweep_config['parameters'])
      config = configs[0]
      experiment_config = setup_experiment_config(
          base_path=base_path,
          group=config.pop('group', args.search),
          config=config,
          datetime_name=True,
      )
      # -------------------
      # load hyra config from experiment config
      # run experiment and optionally remove wandb files
      # -------------------
      config = load_config_fn(
        experiment_config=experiment_config,
        config_path=config_path,
        debug=args.debug,
      )
      # -------------------
      # load wandb args
      # -------------------
      algo_name = config.get('alg', None)
      env_name = config["rlenv"].get("ENV_NAME", 'env')
      project = config["PROJECT"]
      if args.debug:
        project += "_debug"

      wandb_init = dict(
          project=project,
          entity=config['user']["ENTITY"],
          group=experiment_config.get('wandb_group', 'default'),
          name=experiment_config.get('wandb_name', f'{algo_name}_{env_name}'),
          tags=[algo_name.upper(), env_name.upper()],
          config=config,
          save_code=False,
          dir=experiment_config['log_dir'],
      )
      wandb.init(**wandb_init)
      run_fn(
        config=config,
        save_path=experiment_config['log_dir']
      )
      return

  else:
    raise NotImplementedError