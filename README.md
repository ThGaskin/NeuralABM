# Neural ODEs for Multi-Agent models
### Thomas Gaskin

---

This project combines multi-agent models with a neural core, in order to estimate densities on ODE parameters from data. The project 
uses the [utopya package](https://docs.utopia-project.org/html/index.html) to handle simulation configuration, data management, 
and plotting. 

> **_Note_**: This README gives a brief introduction to installation and running a model, as well as a basic 
> overview of the Utopia syntax. You can find a complete guide on running models with Utopia/utopya 
> [here](https://docs.utopia-project.org/html/getting_started/tutorial.html#tutorial).

> **_Note_**: See 'Configuration sets' below for guidance on how to reproduce the plots from the
> publication, once you have completed installation.
> 
## How to install
#### 1. Clone this repository
Clone this repository using a link obtained from 'Code' button (for non-developers, use HTTPS):

```console
git clone <GIT-CLONE-URL>
```

#### 2. Install requirements
We recommend creating a new virtual environment in a location of your choice and installing all requirements into the 
venv. The following command will install the [utopya package](https://gitlab.com/utopia-project/utopya) and the utopya CLI
from [PyPI](https://pypi.org/project/utopya/), as well as all other requirements:

```console
pip install -r requirements.txt
```

This assumes your current directory is the project folder.
You should now be able to invoke the utopya CLI:
```console
utopya --help
```

> **_Note_**  On Apple Silicon devices running macOS 12.3+, follow [these](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/)
> instructions to install pytorch and enable GPU training. Your training device will then be set to 'mps'. 
> On all devices the GPU, where available, will always be the preferred training device.

#### 3. Register the project and all models with utopya

In the project directory (i.e. this one), register the entire project using the following command:
```console
utopya projects register .
```
You should get a positive response from the utopya CLI and your project should appear in the project list when calling:
```console
utopya projects ls
```
> **_Note_** Any changes to the project info file need to be communicated to utopya by calling the registration command anew.
> You will then have to additionally pass the `````--exists-action overwrite````` flag, because a project of that name already exists.
> See ```utopya projects register --help``` for more information.

Finally, register a model via
```console
utopya models register from-manifest path/to/model_info.yml
```
For instance, for the `HarrisWilson` model, this will be 

```console
utopya models register from-manifest models/HarrisWilson/HarrisWilson_info.yml
```
Done! 🎉


#### 4. (Optional) Download the datasets, which are stored using git lfs
There are a number of datasets available, both real and synthetic, you can use in order to test the model.
In order to save space, example datasets have been uploaded using [git lfs](https://git-lfs.github.com) (large file 
storage). To download, first install lfs via
```console
git lfs install
```
This assumes you have the git command line extension installed. Then, from within the repo, do
```console
git lfs pull
```
This will pull all the datasets.

## How to run a model
Now you have set up the model, run it by invoking
```console
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
```console
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
```console
utopya run HarrisWilson path/to/cfg.yml
```

> **_Note_**: The models all come with plenty of example configuration files in the `cfgs` folders. These are
> *configuration sets*, complete sets of run configurations and evaluation routines designed to produce specific
> plots. These also demonstrate how to load datasets to run the models.

## Parameter sweeps
> **_Note_**: Take a look at the [full tutorial entry](https://docs.utopia-project.org/html/getting_started/tutorial.html#parameter-sweeps)
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

```console
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

```console
utopya run HarrisWilson --cfg-set <name_of_cfg_set>
```

> **_Note_** Some of the configuration sets perform *sweeps*, that is, runs over several parameter configurations.
> These may take a while to run. 

Running the configuration set will produce plots. If you wish to re-evaluate a run (perhaps plotting different figures),
you do not need to re-run the model, since the data has already been generated. Simply call

```console
utopya eval HarrisWilson --cfg-set <name_of_cfg_set>
```

This will re-evaluate the *last model you ran*. You can re-evaluate any dataset, of course, by
providing the path to that dataset, like so:

```console
utopya eval HarrisWilson path/to/output/folder --cfg-set <name_of_cfg_set>
```
## How to adjust the neural net configurations
You can vary the size of the neural net and the activation functions
right from the config. The size of the input layer is inferred from 
the data passed to it, and the size of the output layer is 
determined by the number of parameters you wish to learn — all the hidden layers
can be determined by the user. The net is configured from the ``NeuralNet`` key of the
config:

```yaml
NeuralNet:
  num_layers: 6
  nodes_per_layer: 20
  activation_funcs:
    first: sine
    2: cosine
    3: tanh
    last: abs
  bias: True
  init_bias: [0, 4]
  learning_rate: 0.002
```
``num_layers`` and ``nodes_per_layer`` give the structure of the hidden layers (hidden layers 
with different numbers of nodes is not yet supported). The ``activation_funcs`` dictionary
allows specifying the activation function on each layer: just add the number of the layer together 
with the name of a common function, such as ``relu``, ``linear``, ``tanh``, ``sigmoid``, etc.
``bias`` controls use of the bias, and the ``init_bias`` sets the initialisation interval for the 
bias.

## Loading data
See the model-specific README files to see how to load different types of data. Data is stored in the `data/`
folder.

## 🚧 Tests (WIP)
To run tests, invoke
```bash
pytest tests
```
from the main folder.


