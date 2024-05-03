# Install
**load modules**
```
module load Mambaforge/23.11.0-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01
module load cuda/12.2.0-fasrc01
```

**Create and activate conda environment**
```
mamba create -n jaxneurorl python=3.10.9 pip wheel -y
# in case a mamba env is already active
mamba deactivate  # keep running until no env is active
mamba activate jaxneurorl
pip install -r requirements.txt
```
Expected errors:
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
gym3 0.3.3 requires cffi<2.0.0,>=1.13.0, which is not installed.
gym3 0.3.3 requires imageio<3.0.0,>=2.6.0, which is not installed.
flashbax 0.0.1 requires typing-extensions<4.6.0, but you have typing-extensions 4.9.0 which is incompatible.
```

**test that using correct python version (3.10)**
```
python -c "import sys; print(sys.version)"
```
if not, your `path` environment variable/conda environment activation might be giving another system `python` priority.

**Setting up `LD_LIBRARY_PATH`**.
This is important for jax to properly link to cuda. Unfortunately, relatively manual. You'll need to find where your `cudnn` lib is. Mine is at the path below. `find` might be a useful bash command for this.

```
# put cudnn path at beginning of path (before $LD_LIBRARY_PATH)

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/n/sw/helmod-rocky8/apps/Core/cudnn/8.9.2.26_cuda12-fasrc01/lib/

# add conda lib to end of path (after $LD_LIBRARY_PATH)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
```

## Installing JAX
see guide at: https://jax.readthedocs.io/en/latest/installation.html

**pip install:**
```
pip uninstall jax jaxlib -y
pip install -U "jax[cuda12_local]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**test jax install**
```
TF_CPP_MIN_LOG_LEVEL=0 python -c "import jax; print(f'GPUS: {jax.device_count()}'); jax.random.split(jax.random.PRNGKey(42), 2); print('hello world');"
```
Expected is something that includes
```
GPUS: 1
hello world
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
pip install -e '.[qlearning]' "jax==0.4.20"
cd $cur_dir
pip install -r requirements-2.txt
```
Expected errors:
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
gym3 0.3.3 requires cffi<2.0.0,>=1.13.0, which is not installed.
torch 2.0.1 requires nvidia-cublas-cu11==11.10.3.66; platform_system == "Linux" and platform_machine == "x86_64", which is not installed.
torch 2.0.1 requires nvidia-cuda-cupti-cu11==11.7.101; platform_system == "Linux" and platform_machine == "x86_64", which is not installed.
torch 2.0.1 requires nvidia-cuda-runtime-cu11==11.7.99; platform_system == "Linux" and platform_machine == "x86_64", which is not installed.
torch 2.0.1 requires nvidia-cudnn-cu11==8.5.0.96; platform_system == "Linux" and platform_machine == "x86_64", which is not installed.
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
echo 'module load cudnn/8.9.2.26_cuda12-fasrc01' > $activation_dir/env_vars.sh
echo 'module load cuda/12.2.0-fasrc01' >> $activation_dir/env_vars.sh


# setting PYTHONPATH added to activation
echo 'export PYTHONPATH=$PYTHONPATH:`pwd`' >> $activation_dir/env_vars.sh

# setting LD_LIBRARY_PATH added to activation
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' >> $activation_dir/env_vars.sh
echo 'export LD_LIBRARY_PATH=/n/sw/helmod-rocky8/apps/Core/cudnn/8.9.2.26_cuda12-fasrc01/lib/:$LD_LIBRARY_PATH' >> $activation_dir/env_vars.sh

```



## (Optionally) permanently set the results directory
```
echo 'export RL_RESULTS_DIR=${results_dir}' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```
example:
```
echo 'export RL_RESULTS_DIR=$HOME/results/jaxrl_results' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
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
