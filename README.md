# Neural parameter calibration of multi-agent models
### Thomas Gaskin

---

[![CI](https://github.com/ThGaskin/NeuralABM/actions/workflows/pytest.yml/badge.svg)](https://github.com/ThGaskin/NeuralABM/actions/workflows/pytest.yml)
[![coverage badge](https://thgaskin.github.io/NeuralABM/coverage-badge.svg)](https://thgaskin.github.io/NeuralABM)
[![Python 3.6](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

This project combines multi-agent models with a neural core to estimate marginal densities on ODE parameters
(including adjacency matrices) from data. The project
uses the [utopya package](https://docs.utopia-project.org/html/index.html) to handle simulation configuration, data management,
and plotting.

This README gives a brief introduction to installation and running a model, as well as a basic
overview of the Utopia syntax. You can find a complete guide on running models with Utopia/utopya
[here](https://docs.utopia-project.org/html/getting_started/tutorial.html#tutorial).

> [!WARNING]
> This package requires `Python >= 3.11`. utopya is also currently only supported on Unix systems (macOS and Ubuntu).

> [!TIP]
> If you encounter any difficulties, please [file an issue](https://github.com/ThGaskin/NeuralABM/issues/new).

> [!TIP]
> See the section on [configuration sets](#running-a-model-using-configuration-sets) for guidance on how to reproduce the plots from the
> publications, once you have completed installation.
> - T. Gaskin, G. Pavliotis, M. Girolami. *Neural parameter calibration for large-scale multiagent models.* PNAS **120**, 7, 2023.
> https://doi.org/10.1073/pnas.2216415120 (`HarrisWilson` and `SIR` models)
> - T. Gaskin, G. Pavliotis, M. Girolami, . *Inferring networks from time series: a neural approach.* https://arxiv.org/abs/2303.18059
> (`Kuramoto` and `HarrisWilsonNW` models)
> - T. Gaskin, T. Conrad, G. Pavliotis, C. Schütte. *Neural parameter calibration and uncertainty quantification for epidemic
> forecasting*. https://arxiv.org/abs/2312.03147 (`SIR` and `Covid` models)
>
> Since the code is continuously
> being reworked and improved, the plots produced by the current version may quantatively differ from the publication
> plots. Versions below `v2.0.0` will reproduce the PNAS publication plots.

> [!TIP]
> See the model-specific README files, located at ``<model_name>/README.md``, for guidance on 
> how to use individual models.

### Contents of this README
* [How to install](#how-to-install)
* [How to run a model](#how-to-run-a-model)
* [Parameter sweeps](#parameter-sweeps)
* [Running a model using configuration sets](#running-a-model-using-configuration-sets)
* [How to adjust the neural net configuration](#how-to-adjust-the-neural-net-configuration)
* [Training settings](#training-settings)
  * [Changing the loss function](#changing-the-loss-function)
* [Loading data](#loading-data)
* [Tests](#tests)

## How to install
#### 1. Clone this repository
Clone this repository using a link obtained from 'Code' button (for non-developers, use HTTPS):

```commandline
git clone <GIT-CLONE-URL>
```

#### 2. Install requirements
We recommend creating a new virtual environment in a location of your choice and installing all requirements into the
venv. The following command will install the [utopya package](https://gitlab.com/utopia-project/utopya) and the utopya CLI
from [PyPI](https://pypi.org/project/utopya/), as well as all other requirements:

```commandline
pip install -r requirements.txt
```

This assumes your current directory is the project folder.
You should now be able to invoke the utopya CLI:
```commandline
utopya --help
```

> [!NOTE] 
> Enabling CUDA for PyTorch requires additional packages, e.g. `torchvision` and `torchaudio`.
> Follow [these](https://pytorch.org/get-started/locally/) instructions to enable GPU training.
> For Apple Silicon, follow [these](https://PyTorch.org/blog/introducing-accelerated-pytorch-training-on-mac/)
> installation instructions. Note that GPU acceleration for Apple Silicon is still work in progress and many functions have not
> yet been implemented.

#### 3. Register the project and all models with utopya

In the project directory (i.e. this one), register the entire project and all its models using the following command:
```commandline
utopya projects register . --with-models
```
You should get a positive response from the utopya CLI and your project should appear in the project list when calling:
```commandline
utopya projects ls
```
Done! 🎉

> [!IMPORTANT]
> Any changes to the project info file need to be communicated to utopya by calling the registration command anew.
> You will then have to additionally pass the `````--exists-action overwrite````` flag, because a project of that name already exists.
> See ```utopya projects register --help``` for more information.

#### 4. (Optional, but recommended) Install latex
To properly display mathematical equations and symbols in the plots, we recommend installing latex. However, latex distributions
are typically quite large, so ensure you have enough space on your disk.

On Ubuntu, first install latex by running
```commandline
sudo apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super
```
For macOS, install latex via a package manager, e.g. Homebrew or ports.

For both operating systems, also run the following command from within the virtual environment:
```commandline
pip install latex
```
Thereafter, set the plots to use latex by changing the following entry in the `base_plots.yaml` file of the model:
```yaml
.default_style:
  style:
    text.usetex: True
  # Keep everything else unchanged
```
Latex will then be used in *all* model plots. You can also change this individually for each plot.

#### 5. (Optional) Download the datasets, which are stored using git lfs
There are a number of datasets available, both real and synthetic, you can use in order to test the model.
In order to save space, example datasets have been uploaded using [git lfs](https://git-lfs.github.com) (large file
storage). To download, first install lfs via
```commandline
git lfs install
```
This assumes you have the git command line extension installed. Then, from within the repo, do
```commandline
git lfs pull
```
This will pull all the datasets.

## How to run a model
Now you have set up the model, run it by invoking
```commandline
utopya run model_name
```
We will be using the `HarrisWilson` model as an example in the following, so simply replace all instances of
``HarrisWilson`` with your own model name.

The `HarrisWilson` model will generate a synthetic dataset of economic flow on a network, train the neural net for 1000 epochs
(default value), and write the output into a new directory, located in your home directory
`~/utopya_output` by default.

The default configuration settings are provided in the `HarrisWilson_cfg.yml` file in the
`models/HarrisWilson` folder. You can modify the settings here, but we recommend changing the configuration
settings by instead creating a `run.yml` file somewhere and using it to run the model. You can do so by
calling
```commandline
utopya run HarrisWilson path/to/run_cfg.yml
```
In this file, you only need to specify those entries from the `<modelname>_cfg.yml` file you wish to change,
and not reproduce the entire configuration set. The advantage of this approach is that you can
create multiple configs for different scenarios, and leave the working base configuration untouched.
An example could look like this:

```yaml
parameter_space:
  seed: 4
  num_epochs: 3000
  write_start: 1
  write_every: 1
  HarrisWilson:
    Data:
      synthetic_data:
        alpha: 1.5
        beta: 4.2
        sigma: 0.1
```
This is generating a synthetic dataset using all the settings from the `HarrisWilson_cfg.yml` file *except* for those
You can run the model using this file by calling
```commandline
utopya run HarrisWilson path/to/cfg.yml
```

> [!TIP] 
> The models all come with plenty of example configuration files in the `cfgs` folders. These are
> *configuration sets*, complete sets of run configurations and evaluation routines designed to produce specific
> plots. These also demonstrate how to load datasets to run the models.

## Parameter sweeps

> [!TIP]
> Take a look at the [full tutorial entry](https://docs.utopia-project.org/html/getting_started/tutorial.html#parameter-sweeps)
> for a full guide on running parameter sweeps.

Parameter sweeps (multiple runs using different configuration settings) are easy: all you need to do is add a
`!sweep` tag to all parameters you wish to sweep over. Parameter sweeps are automatically run in parallel.
For example, to sweep over the `seed` (to generate some statistics, say), just do

```yaml
parameter_space:
  seed: !sweep
    default: 0
    range: [10]
```
Then call your model via

```commandline
utopya run <model_name> --run-mode sweep
```
The model will then run ten times, each time using a different seed value. You can also add the following entry to
the configuration file at the root-level:
```yaml
perform_sweep: True
```
You can then run a sweep without the ``--run-mode`` flag in the CLI.
Passing a `default` argument to the sweep parameter(s) is required: this way, the model can still perform a single run
when a sweep is not configured. Again, there are plenty of examples in the `cfgs` folders.


## Running a model using configuration sets
Configuration sets are a useful way of gathering a combination of run settings and plot configurations
in a single place, so as to automatically generate data and plots that form a set.
The `HarrisWilson` model contains a large number of *configuration sets* comprising run configs and *evaluation* configs,
that is, plot configurations. These sets will reproduce the plots from the publication.
You can run them by executing

```commandline
utopya run HarrisWilson --cfg-set <name_of_cfg_set>
```

> [!NOTE]
> Some of the configuration sets perform *sweeps*, that is, runs over several parameter configurations.
> These may take a while to run.

Running the configuration set will produce plots. If you wish to re-evaluate a run (perhaps plotting different figures),
you do not need to re-run the model, since the data has already been generated. Simply call

```commandline
utopya eval HarrisWilson --cfg-set <name_of_cfg_set>
```

This will re-evaluate the *last model you ran*. You can re-evaluate any dataset, of course, by
providing the path to that dataset, like so:

```commandline
utopya eval HarrisWilson path/to/output/folder --cfg-set <name_of_cfg_set>
```
## How to adjust the neural net configuration
### General architecture
You can vary the size of the neural net and the activation functions
right from the config. The size of the input layer is inferred from
the data passed to it, and the size of the output layer is
determined by the number of parameters you wish to learn — all the hidden layers
can be determined by the user. The net is configured from the ``NeuralNet`` key of the
config:

```yaml
NeuralNet:
  num_layers: 6
  nodes_per_layer:
    default: 20
    layer_specific:
      0: 10
  activation_funcs:
    default: sigmoid
    layer_specific:
      0: sine
      1: cosine
      2: tanh
      -1: abs
  biases:
    default: [0, 4]
    layer_specific:
      1: [-1, 1]
  learning_rate: 0.002
```
``num_layers`` sets the number of hidden layers. ``nodes_per_layer``, ``activation_funcs``, and ``biases`` are
dictionaries controlling the structure of the hidden layers. Each requires a ``default`` key
giving the default value, applied to all layers. An optional ``layer_specific`` entry
controls any deviations from the default on specific layers; in the above example,
all layers have 20 nodes by default, use a sigmoid activation function, and have a bias
which is initialised uniformly at random on [0, 4]. Layer-specific settings are then provided.
You can also set the bias initialisation interval to `default`: this will initialise the bias using the [PyTorch default](https://github.com/pytorch/pytorch/blob/9a575e77ca8a0be7a3f3625c4dfdc6321d2a0c2d/torch/nn/modules/linear.py#L72)
Xavier uniform distribution.

### Activation functions
Any [PyTorch activation function](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)
is supported, such as ``relu``, ``linear``, ``tanh``, ``sigmoid``, etc. Some activation functions take arguments and
keyword arguments; these can be provided like this:

```yaml
NeuralNet:
  num_layers: 6
  nodes_per_layer: 20
  activation_funcs:
    default:
      name: Hardtanh
      args:
        - -2 # min_value
        - +2 # max_value
      kwargs:
        # any kwargs here ...
```

### Specifying the prior on the output
For many applications, you will want control over the prior distribution of the parameters. To this
end, you can add a `prior` entry that gives a distribution over the parameters you wish to learn:
```yaml
NeuralNet:
  prior:
    distribution: uniform
    parameters:
      lower: 0
      upper: 2
```
This will train the neural network to initially output values uniformly within [0, 2], for all
parameters you wish to learn. If you want individual parameters to have their own priors, you can do so by passing a
list as the argument to `prior`. For instance, assume you wish to learn 2 parameters; the configuration entry then could
be:
```yaml
NeuralNet:
  prior:
    - distribution: normal
      parameters:
        mean: 0.5
        std: 0.1
    - distribution: uniform
      parameters:
        lower: 0
        upper: 5
```
This will initialise each parameter with a separate prior.
## Training settings
You can modify the training settings, such as the batch size or the training device, from the
`Training` entry of the config:

```yaml
Training:
  batch_size: 1
  loss_function:
    name: MSELoss
  to_learn: [ param1, param2, param3 ]
  true_parameters:
    param4: 0.5
  device: cpu
  num_threads: ~
```
The `to_learn` entry lists the parameters you wish to learn. If you are not learning the complete
parameter set, you must supply the parameter value to use during training for that parameter under
`true_parameters`.

The `device` entry sets the training device. The default here is the `cpu`; you can set it to any
supported PyTorch training device; for instance, set it to `cuda` to use the GPU for training. Make sure your platform
is configured to support the selected device.
On Apple Silicon, set the device to `mps` to enable GPU training, provided you have followed the corresponding
installation instructions (see above). Note that PyTorch for Apple Silicon is still work in progress at this stage,
and some functions have not yet been fully implemented.

`utopya` automatically parallelises multiple runs; the number of CPU cores available to do this
can be specified under `worker_managers/num_workers` on the root-level configuration (i.e. on the same level as
`parameter_space`). The `Training/num_threads` entry controls the number of threads *per model run* to be used during training.
If you thus set `num_workers` to 4 and `num_threads` to 3, you will in total be able to use 12 threads.

### Changing the loss function
You can set the ``loss_function/name`` argument to point to any supported
[Pytorch loss function](https://pytorch.org/docs/stable/nn.html#loss-functions). Additional arguments to
the loss function can be passed via an optional ``args`` and ``kwargs`` entry:

```yaml
loss_function:
  name: CTCLoss
  args:
    - 1  # blank
    - 'sum' # reduction to use
```
## Loading data
See the model-specific README files to see how to load different types of data. Data is stored in the `data/`
folder.

## Tests
The models and core functionalities (graph generation, neural network settings,
vector, etc.) are unit tested. To run the tests, invoke
```commandline
pytest tests/
```
from the main folder. This will run all tests; to output a coverage report, invoke
```commandline
coverage run -m pytest tests/ && coverage report -m
```
