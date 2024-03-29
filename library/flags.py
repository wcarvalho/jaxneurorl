from absl import flags

# flags.DEFINE_integer('num_actors', 6, 'number of actors.')

# WANDB
flags.DEFINE_bool('wandb', True, 'whether to use wandb.')
flags.DEFINE_string('entity', '', 'wandb entity.')

# INDIVIDUAL RUN
flags.DEFINE_string('search', '', 'which search to use.')
flags.DEFINE_string(
    'parallel', 'none', "none: run 1 experiment. sbatch: run many experiments with SBATCH. ray: run many experiments with say. use sbatch with SLUM or ray otherwise.")
flags.DEFINE_bool(
    'debug', False, 'whether to run in debug mode.')

# SLURM
flags.DEFINE_integer('config_idx', 1, 'number of actors.')
flags.DEFINE_integer('num_cpus', 16, 'number of cpus.')
flags.DEFINE_integer('memory', 120_000, 'memory (in mbs).')
flags.DEFINE_integer('max_concurrent', 12, 'number of concurrent jobs')

flags.DEFINE_string('account', '', 'account on slurm servers to use.')
flags.DEFINE_string('partition', 'kempner', 'account on slurm servers to use.')
flags.DEFINE_string('time', '0-01:00:00', '1 hour.')

flags.DEFINE_integer('num_gpus', 1, 'number of gpus.')
# flags.DEFINE_bool('skip', False, 'whether to skip experiments that have already run.')
flags.DEFINE_bool('debug_parallel', False, 'whether to debug parallel runs.')

# DO NOT USE BELOW. Set by parallel.py
flags.DEFINE_string('search_config', '', 'config file produced by parallel')
flags.DEFINE_bool('subprocess', False, 'label for whether this run is a subprocess.')