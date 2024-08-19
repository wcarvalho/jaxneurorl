import wandb
import os
from multiprocessing import Pool
import yaml
from wandb_search import run_agent
import argparse

def run_wandb_sweep_on_gpus(sweep_config, project_name, entity, gpus):
    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project=project_name, entity=entity)
    print(f"Sweep ID: {sweep_id}")

    p = Pool(len(gpus))
    args = [(sweep_id, gpu) for gpu in gpus]
    p.map(run_agent, args)

def parse_gpu_list(gpu_string):
    return [int(gpu.strip()) for gpu in gpu_string.split(',')]

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run WandB sweep on multiple GPUs")
    parser.add_argument("--gpus", type=str, required=True, help="Comma-separated list of GPU ids to use")
    parser.add_argument("--project", type=str, required=True, help="WandB project name")
    parser.add_argument("--entity", type=str, required=True, help="WandB entity name")
    parser.add_argument("--config", type=str, required=True, help="Path to the sweep config YAML file")

    args = parser.parse_args()

    # Load sweep config
    with open(args.config, 'r') as f:
        sweep_config = yaml.safe_load(f)

    # Parse GPU list
    gpus = parse_gpu_list(args.gpus)

    run_wandb_sweep_on_gpus(sweep_config, args.project, args.entity, gpus)