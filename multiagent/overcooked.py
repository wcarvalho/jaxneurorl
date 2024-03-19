"""

TESTING:
python -m ipdb -c continue multiagent/overcooked.py \
  --debug=True \
  --search=default

TESTING SLURM LAUNCH:
python multiagent/overcooked.py \
  --parallel=sbatch \
  --debug_parallel=True \
  --search=default

RUNNING ON SLURM:
python multiagent/overcooked.py \
  --parallel=sbatch \
  --search=default
"""

from absl import flags
from absl import app
# from absl import logging


import os
import jax
from typing import NamedTuple, Dict, Union

from ray import tune

import wandb
from omegaconf import OmegaConf
from safetensors.flax import save_file
from flax.traverse_util import flatten_dict

from jaxmarl import make
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.environments.overcooked import overcooked_layouts


os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import library.flags
from library import parallel
from multiagent import iql
FLAGS = flags.FLAGS

def run_single(
        sweep_config: dict,
        config_path: str,
        save_path: str = None):

    config, wandb_init = parallel.load_hydra_config(
        sweep_config,
        config_path=config_path,
        save_path=save_path,
        tags=[f"jax_{jax.__version__}"]
        )
    wandb.init(**wandb_init)

    config['env']["ENV_KWARGS"]["layout"] = overcooked_layouts[config['env']["ENV_KWARGS"]["layout"]]
    env = make(config["env"]["ENV_NAME"], **config['env']['ENV_KWARGS'])
    env = LogWrapper(env)

    alg_name = config['alg']
    if alg_name == 'iql':
      make_train = iql.make_train
    else:
      raise NotImplementedError(alg_name)

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config, env)))
    outs = jax.block_until_ready(train_vjit(rngs))

    if save_path is not None:
        def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
            flattened_dict = flatten_dict(params, sep=',')
            save_file(flattened_dict, filename)

        model_state = outs['runner_state'][0]
        params = jax.tree_map(lambda x: x[0], model_state.params) # save only params of the firt run
        os.makedirs(save_path, exist_ok=True)
        save_params(params, f'{save_path}/{alg_name}.safetensors')
        print(f'Parameters of first batch saved in {save_path}/{alg_name}.safetensors')

def sweep(search: str = ''):
  search = search or 'default'

  if search == 'default':
    space = [
        {
            "group": tune.grid_search(['run-3-iql']),
            "alg": tune.grid_search(['iql']),
        }
    ]
  else:
    raise NotImplementedError(search)

  return space

def main(_):
  parallel.run(
      trainer_filename=__file__,
      config_path='../configs',  # must be relative...
      run_fn=run_single,
      sweep_fn=sweep,
      folder=os.environ.get(
          'RL_RESULTS_DIR', '/tmp/rl_results_dir')
  )

if __name__ == '__main__':
  app.run(main)