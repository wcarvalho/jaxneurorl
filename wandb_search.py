import wandb
import subprocess
import os
import time
from concurrent.futures import ThreadPoolExecutor
import jax
from functools import partial
import yaml

# Assuming run_single and other necessary imports are available
from singleagent.baselines import run_single  # Make sure to import this correctly

def run_sweep():
    wandb.init(project="jax-neurorl-baselines")
    base_config = None
    with open("configs/qlearning_sumit.yaml") as f:
        base_config = yaml.load(f, Loader=yaml.FullLoader)
    base_config['ENV_NAME'] = 'Craftax-Symbolic-v1'
    config = base_config.copy()
    run_config = wandb.config
    config.update(run_config)
    config["alg"] = "qlearning"
    run_single(config)

# The following function starts a sweep. Use this function if starting sweeps
# programatically instead of using the CLI
def run_agent(sweep_gpu_id):
    sweep_id, gpu_id = sweep_gpu_id
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    wandb.agent(sweep_id, function=run_sweep)

if __name__ == "__main__":
    run_sweep()