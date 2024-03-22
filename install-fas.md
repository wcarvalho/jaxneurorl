# Install
**load modules**
```
module load cuda/12.2.0-fasrc01
```

**Create and activate conda environment**
```
mamba create -n jaxneurorl python=3.9 pip wheel -y
mamba env update -f conda_env.yml
mamba deactivate  # in case a mamba env is already active
mamba activate jaxneurorl
```

**test that using correct python version (3.9)**
```
python -c "import sys; print(sys.version)"
```
if not, your `path` environment variable/conda environment activation might be giving another system `python` priority.

**Setting up `LD_LIBRARY_PATH`**.
This is important for jax to properly link to cuda. Unfortunately, relatively manual. You'll need to find where your `cudnn` lib is. Mine is at the path below. `find` might be a useful bash command for this.

```
# put cudnn path at beginning of path (before $LD_LIBRARY_PATH)
unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/n/sw/helmod-rocky8/apps/Core/cudnn/8.9.2.26_cuda12-fasrc01/lib/:$LD_LIBRARY_PATH

# add conda lib to end of path (after $LD_LIBRARY_PATH)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
```

## Installing JAX
see guide at: https://jax.readthedocs.io/en/latest/installation.html

**pip install:**
```
pip install -U "jax[cuda12_local]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**test jax install**
```
TF_CPP_MIN_LOG_LEVEL=0 python -c "import jax; print(f'GPUS: {jax.device_count()}'); jax.random.split(jax.random.PRNGKey(42), 2); print('hello world');"
```

## Installing other libraries

**JaxMARL**
```
# store current directory
cur_dir=`pwd`

# make install directory inside conda env path
jaxmarl_loc=$CONDA_PREFIX/github/jaxmarl
mkdir -p $jaxmarl_loc
git clone https://github.com/FLAIROx/JaxMARL.git $jaxmarl_loc

# install jaxmarl
cd $jaxmarl_loc
git checkout cc9f12bb5948c31c478a1d662c56a8d7c5f8c530
pip install -e '.[qlearning]'
cd $curdir
```

**notes**: if you're using IntelliSense (e.g. through vscode), you'll need to add the jaxmarl path to `python.autoComplete.extraPaths`. you can access it with `echo $jaxmarl_loc`

# Setup conda activate/deactivate

**ONE TIME CHANGE TO MAKE YOUR LIFE EASIER**. if you want to avoid having to load modules and set environment variables each time you load this environment, you can add loading things to the activation file. Below is how.

```
# first activate env
mamba activate jaxneurorl

# make activation/deactivation directories
activation_dir=$CONDA_PREFIX/etc/conda/activate.d
mkdir -p $activation_dir
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

# module loading added to activation
echo 'module load cuda/12.2.0-fasrc01' > $activation_dir/env_vars.sh

# setting PYTHONPATH added to activation
echo 'export PYTHONPATH=$PYTHONPATH:.' >> $activation_dir/env_vars.sh
# below makes jaxmarl visible to IDE-like functionality
echo 'export PYTHONPATH=$PYTHONPATH:$jaxmarl_loc' >> $activation_dir/env_vars.sh

# setting LD_LIBRARY_PATH added to activation
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' >> $activation_dir/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/n/sw/helmod-rocky8/apps/Core/cudnn/8.9.2.26_cuda12-fasrc01/lib/' >> $activation_dir/env_vars.sh

# undoing LD_LIBRARY_PATH added to deactivation
echo 'unset LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
```



## (Optionally) permanently set the results directory
```
echo 'export RL_RESULTS_DIR=${results_dir}' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```
example:
```
echo 'export RL_RESULTS_DIR=$HOME/jaxrl_results' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

Otherwise, can set each time run experiment
```
RL_RESULTS_DIR=${results_dir} python trainer.py
```

## (Optional) setup wandb
```
wandb login
```
Once you set up a wandb project and have logged runs, group them by the following settings:
- Group: name of search run
- Name: name of individual runs (this aggregates all seeds together)
