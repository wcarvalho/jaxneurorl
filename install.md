# Install

**Recommendation:** Install [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)

**Create and activate conda environment**
```
mamba create -n jaxneurorl python=3.9 pip wheel -y
mamba env update -f conda_env.yaml
# in case a mamba env is already active
mamba deactivate
mamba activate jaxneurorl
```

**test that using correct python version (3.9)**
```
python -c "import sys; print(sys.version)"
```
if not, your `path` environment variable/conda environment activation might be giving another system `python` priority.

## Installing JAX
see guide at: https://jax.readthedocs.io/en/latest/installation.html

**pip install:**
```
pip install -U "jax==0.4.20"  "jaxlib==0.4.20"
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
