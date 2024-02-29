# Neural parameter calibration of ODEs and SDEs
### Thomas Gaskin

---

[![CI](https://github.com/ThGaskin/NeuralABM/actions/workflows/pytest.yml/badge.svg)](https://github.com/ThGaskin/NeuralABM/actions/workflows/pytest.yml)
[![coverage badge](https://thgaskin.github.io/NeuralABM/coverage-badge.svg)](https://thgaskin.github.io/NeuralABM)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

<img src="https://github.com/ThGaskin/NeuralABM/assets/22022754/e0bf61c7-4fe1-4234-b480-02d1f8efff6b" width=49%> <img src="https://github.com/ThGaskin/NeuralABM/files/13863262/marginals_all.pdf" width=49%> 

<img src="https://github.com/ThGaskin/NeuralABM/files/13863293/densities_from_joint.pdf" width=49%> <img src="https://github.com/ThGaskin/NeuralABM/files/13863249/predictions.pdf" width=49%>


This project calibrates multi-agent ODE and SDE models to data using a neural network. We estimate marginal densities on the equation parameters, including adjacency matrices. This repository contains all the code and models used in our publications on the topic, as well as an extensive set of tools and examples for you to calibrate your own model:

- T. Gaskin, G. Pavliotis, M. Girolami. *Neural parameter calibration for large-scale multiagent models.* PNAS **120**, 7, 2023.
https://doi.org/10.1073/pnas.2216415120 (`HarrisWilson` and `SIR` models)
- T. Gaskin, G. Pavliotis, M. Girolami, . *Inferring networks from time series: a neural approach.* https://arxiv.org/abs/2303.18059
(`Kuramoto` and `HarrisWilsonNW` models)
- T. Gaskin, T. Conrad, G. Pavliotis, C. Schütte. *Neural parameter calibration and uncertainty quantification for epidemic
forecasting*. https://arxiv.org/abs/2312.03147 (`SIR` and `Covid` models)

Each model contains its own README file, detailing specifics on the code for the different models. Since the code is continuously being reworked and improved, the plots produced by the current version may differ from the publication plots. For this reason, this repository is versioned, such that each publication has a version that will produce the plots exactly as they appear in the paper. However, the results produced by the latest code base will typically be more accurate, performative, and reliable than older versions.

The project uses the [utopya package](https://docs.utopia-project.org/html/index.html) to handle simulation configuration, data management,
and plotting. This README gives a brief introduction to installation and a basic tutorial, which will be sufficient to just run the models, reproduce plots, and play around with the code. You can also refer to the model-specific README files, located at `<model_name>/README.md`, for detailed instructions on each model's features. A complete guide to running models with Utopia/utopya can be found [here](https://docs.utopia-project.org/html/getting_started/tutorial.html#tutorial). As you go through the [Tutorial](#tutorial) below, you will find links to the relevant tutorial entries, and it is recommended to peruse these if you wish to build your own model using our code base.

> [!TIP]
> If you encounter any difficulties or have questions, please [file an issue](https://github.com/ThGaskin/NeuralABM/issues/new).


### Contents of this README
* [How to install](#installation)
  * [Installation on Windows](#installation-on-windows) 
* [Tutorial](#tutorial)
  * [How to run a model](#how-to-run-a-model)
  * [Parameter sweeps](#parameter-sweeps)
  * [Evaluation and plotting](#evaluation-and-plotting)
  * [Adjusting the neural net configuration](#adjusting-the-neural-net-configuration)
  * [Training settings](#training-settings)
  * [Changing the loss function](#changing-the-loss-function)
  * [Loading data](#loading-data)
* [Models overview](#models-overview)
* [Building your own model](#building-your-own-model)

---
# Installation
> [!WARNING]
> utopya is currently only fully supported on Unix systems (macOS and Ubuntu). For Windows
> installation instructions, see below; be aware that utopya for Windows is currently work in progress.

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

### Installation on Windows

On Windows systems, you must use the Windows development branch of utopya; after completing the steps above, run:

```commandline
pip uninstall utopya
pip install git+https://gitlab.com/utopia-project/utopya@89-allow-exec-prefix
```

Be aware that development on the utopya Windows dev branch is ongoing; if you run into any problems, please file an [issue](https://gitlab.com/utopia-project/utopya/-/issues/new). 

Lastly, you must change the default encoding to utf-8 on Windows; in the Control Panel, navigate to the 
Regional Settings, go to the 'Administrative' tab, click 'Change system locale' under 'Language for non-Unicode programs',
and check the 'Beta: Use Unicode UTF-8 for worldwide language support option'. See [here](https://stackoverflow.com/questions/57131654/using-utf-8-encoding-chcp-65001-in-command-prompt-windows-powershell-window/57134096#57134096)
for instructions.

---
# Tutorial 
> [!TIP]
> At any stage and for any command, you can use the `--help` flag to show a description of the command, syntax details, and valid arguments, e.g.
> ```commandline
> utopya eval --help
> ```

## How to run a model
Now you have set up the repository, let's run a model. We'll use the `SIR` model as an example. Running a model is a simple command:
```commandline
utopya run SIR
```
You can call 
```commandline
utopya models ls
```
to see a full list of all the registered models. Replace `SIR` with any of the registered model names to run that model
instead.

For all models, this command will generate some synthetic data, train the neural net to calibrate the model equations on it, and generate a series of plots in the 
`utopya_output` directory, located by default in your home directory (but this can be [changed](#changing-the-output-directory)). Once everything is done, you should see an output like this in your terminal:

```commandline
SUCCESS  logging           Performed plots from 5 plot configurations in 37.5s.

SUCCESS  logging           All done.
```

> [!TIP]
> If you get the following error message
> ```commandline
> ValueError: The writer 'ffmpeg' is not available on your system! 
> ```
> you don't have a writer installed to save animations. Don't worry: it's only needed for producing animated plots,
> so the error isn't critical and doesn't prevent you from plotting non-animated plots.

Navigate to your `utopya_output` directory and open the `SIR` folder. In it you should see a time-stamped folder
containing a `config`, a `data`, and an `eval` folder. One of the most important benefits of using utopya is that it automatically
stores data, plots, and all the configuration files used to generate them in a unique folder, and outputs are never overwritten. This makes reproducing
and repeating runs easy, and keeps all the data organised. We will shortly see how you can easily re-evaluate the data 
from a given run without having to re-run the simulation.

This directory structure already hints at the three basic steps that are executed during a model run:

- Combine different configurations, prepare the simulation run(s) and start them.
- Store the data
- Read in the data and automatically evaluate it by calling plot functions.

Open the `eval` folder — in it there will be a further time-stamped folder. Every time you evaluate a simulation, a new folder is created. This way, no evaluation result is ever overwritten. In the `eval/YYMMDD-hhmmss` folder, you should find five plots. Take a look at `densities_from_joint.pdf`, which should look something like this:

<img src="https://github.com/ThGaskin/NeuralABM/files/13787239/densities_from_joint.pdf" width=100%>

You can see the true data (orange) together with the neural net predictions (blue) and an error estimate (blue shaded area).
The results aren't great; you will also notice from the `loss.pdf` plot that the training loss has barely decreased. Why? Well, 
take a look at the `SIR_cfg.yml` file. This file holds all the default parameters for the model run. Scroll down to the `Training` entry: you will notice the `batch_size` is set to 1. This means that the neural network performs a gradient descent step every time it has reproduced a single frame of the time series. Further above, you will notice that the synthetic dataset used to train the model has a length of `num_steps: 100`. For these disease dynamics, let's see if letting the neural network see the whole time series for each gradient descent step would improve things. You could change the batch size in the `SIR_cfg.yml` file directly, but actually this is not recommended: this file holds all the default values the model will fall back on, should something go wrong. Instead create a new `run.yml` file, somewhere on your computer, and copy the following entries into it:

```yaml
parameter_space:
  num_epochs: 300
  SIR:
    Training:
      batch_size: 100
```
We are now using a batch size of 100, i.e. the length of the time series, and are also training the model for a little bit longer (300 epochs instead of the default 100). Now, run the model again and pass the path to this file to the model:

```commandline
utopya run SIR path/to/run.yml
```
Here, we are *only* updating those entries of the base configuration which are also given in the `run.yml` file; the remaining ones are taken from the default configuration file. The results in the output folder should look something like this:

<img src="https://github.com/ThGaskin/NeuralABM/files/13787372/densities_from_joint.pdf" width=100%>

Perhaps a little bit better, but still not great, and the uncertainty is much too small. The real problem here is that we are only training our neural network from a single initialisation, and letting it find one of the possible parameters that fit the problem. This doesn't give us an accurate representation of the parameter space. What we really need to be doing is training it multiple times, in parallel, from different initialisations, so that it can see the more of the parameter space. This is what we will do in the next section.

> [!TIP]
> #### Changing the output directory
> If you wish to save the model output to a different directory, add the following entry to your run configuration:
> ```yaml
> paths:
>   out_dir: ... # path/to/dir
> ```
> or run the model with 
> ```commandline
> utopya run <model_name> -p paths.out_dir path/to/out_dir
> ```

## Parameter sweeps

Take a look at the `models/SIR/cfgs` folder. In it you will find lots of subfolders, each containing a pair of `run.yml` and `eval.yml` files. These are called *configuration sets*: pre-fabricated run files and corresponding evaluation configurations. Try running the following command:

```commandline
utopya run SIR --cs Predictions_from_ABM_data
```
The `--cs` ('configuration set') command tells utopya to use the `run.yml` and later the `eval.yml` file for the plotting routine (we will get to the plots [a little later on](#evaluation-and-plotting)). In the `run.yml` file, take note of the following entries:

```yaml
perform_sweep: True
parameter_space:
  seed: !sweep
    default: 1
    range: [60]
```
The `seed` entry controls the random initialisation of the neural network, and we are 'sweeping' over 60 different initialisations (`range: [60]`) and training the model on the same dataset each time! The `perform_sweep` entry tells the model to run the sweep – set it to `False` to just perform a single run again. The `seed` would then be set to its `default` value, in this case 1. utopya will automatically parallelise the runs over as many cores as your computer makes available (you can [change](#adjusting-the-parallelisation) how many workers it can use). A single run is called a 'universe' run, a sweep run over many 'universes' is called a 'multiverse' run.

Once the run is complete, the plot output should look like this:

<img src="https://github.com/ThGaskin/NeuralABM/files/13787421/densities_from_joint.pdf" width=100%>

Much better! You can see that the predicted densities are significantly closer to the true data. The folder also contains the marginal densities on the parameters we are estimating:

<img src="https://github.com/ThGaskin/NeuralABM/files/13787439/marginals.pdf" width=100%>

These too look good: we obtain an infection parameter of about 0.21, and infection period of about 15 days – these are very similar to the values of 0.2 and 14 used to generate the synthetic data.

> [!TIP]
> You can also configure sweeps by adding a `--run-mode sweep` or `--run-mode single` flag to the command in the CLI:
> ```commandline
> utopya run SIR --run-mode sweep`
> ```
> This will overwrite the settings in the configuration file. In general, paths to `run.yml` files will overwrite the default entries, and CLI flags will overwrite the
> entries in the config file. You can also change parameters right from the CLI:
> ```commandline
> utopya run SIR --pp num_epochs=300
> ```
> See [here](https://docs.utopia-project.org/html/usage/run/config.html) for details. 

In your output folder you will also find the following plot:

<img src="https://github.com/ThGaskin/NeuralABM/files/13852798/predictions.pdf" width=100%>

Each line represents a trajectory taken by the neural net during training; as you can see, we are training the net multiple times in parallel, each time initialising the neural network at a different value of the initial distribution – see [the corresponding section](#specifying-the-prior-on-the-output) on how to adjust this distribution. The colour of each line repressents the training loss at that sample.
The number of initialisations is controlled by the `seed` entry of the run config.

> [!TIP]
> As an exercise, play around with the `seed.range` argument of the `run.yml` config. How does the quality of the time series prediction and marginal densities change as you increase or decrease the number of runs?

### Sweep configurations and multiple sweeps
You can sweep over as many parameters and entries as you like; any key in the run configuration can be swept over. An sweep entry must take the following form:
```yaml
parameter: !sweep
   default: 0
   values: [1, 2, 3, 4]
```
Any configuration file must be compatible with *both* a multiverse ('sweep') and a universe ('single') run. The `default` entry is used whenever a universe run is performed, 
the `values` entry used for the sweep. Instead of specifying a list of `values`, you can also provide a `range`, a `linspace`, or a `logspace`:
```yaml
parameter: !sweep
   default: default_value
   range: [1, 4] # passed to python range()
                 # Other ways to specify sweep values:
                 #   values: [1,2,3,4]  # taken as they are
                 #   range: [1, 4]      # passed to python range()
                 #   linspace: [1,4,4]  # passed to np.linspace
                 #   logspace: [-5, -2, 7]  # 7 log-spaced values in [10^-5, 10^-2], passed to np.logspace
```

Once you have set up your sweep configuration file, enable a multiverse run either by setting `perform_sweep: True` to the top-level of the file, or by passing `--run-mode sweep` to the CLI command when you run your model. Without one of these, the model will be run as a universe run.

There is no limit to how many parameters you can sweep over. Take a look, for instance, at the `models/HarrisWilson/cfgs/Marginals_over_noise/run.yml` file. Here, we are sweeping over the noise in the training data (`sigma`) as well as the `seed`. Sweeping over more parameters takes longer, of course, since the volume of parameters increases exponentially.

> [!TIP]
> Read the full guide on running parameter sweeps [here](https://docs.utopia-project.org/html/getting_started/tutorial.html#parameter-sweeps).

### Coupled sweeps
If you want to sweep over one parameter but vary some others along with it, you can perform a [coupled sweep](https://docs.utopia-project.org/html/about/features.html?highlight=target_name#id31):
```yaml
param1: !sweep
  default: 1
  values: [1, 2, 3, 4]
param2: !coupled-sweep
  default: foo
  values: [bar, baz, foo, fab]
  target_name: param1
```
Here, `param2` is being varied along `param1` – the dimension of the parameter space remains 1. You can couple as many parameters to sweep parameters as you like.

### Adjusting the parallelisation
When running a sweep, you will see the following logging entry in your terminal:
```commandline
PROGRESS logging           Initializing WorkerManager ...
NOTE     logging             Number of available CPUs:  8
NOTE     logging             Number of workers:         8
NOTE     logging             Non-zero exit handling:    raise
PROGRESS logging           Initialized WorkerManager.
```

As you can see, here utopya is using 8 CPU cores as individual workers to run universes in parallel. If you wish to adjust this, e.g. to reduce the load on the CPU, you can adjust the `worker_manager` settings in your configuration file:

```yaml
worker_manager:
  num_workers: 4
```

## YAML configuration files and changing the parameters
As you have seen, there are multiple configuration layers that are recursively updated: at the bottom, there are default configuration entries for each model, stored in `<model_name>_cfg.yml`. These are default values that will, broadly speaking, be useful in most situations. For this reason, it is best to not change them when performaing 
a specific run. The default configuration file should include *all* the defaults used for a model, but you wouldn't want to have to copy-paste *all* of them into a new file if you only want to change a few. For this purpose there are *run-specific* configuration files, which you can pass to the model CLI via 
```commandline
utopya run <model> path/to/run.yml
```
You can pass a relative or an absolute path, it's up to you. Entries in these files will overwrite the default values. Remember that you only need to provide those entries of the default config you wish to update! Finally, you can also change parameters directly by passing a `--pp` flag from the CLI:
```commandline
utopya run <model> --pp num_epochs=100 --pp entry.key=2
```

Note that, when using the CLI, you can set sublevel entries of outer scopes by connecting them with a `.`: `key.subkey.subsubkey`. YAML offers a wide range of functionality within the configuration file. Take a look e.g. at the [learnXinYminutes](https://learnxinyminutes.com/docs/yaml/) YAMl tutorial for an overview – but since it is an intuitive and humand-readable configuration language, most things should seem very familiar to you already.

> [!IMPORTANT]
> YAML is sensitive to indentation levels! In utopya, nearly every option can be set through a configuration parameter. With these, it is important to take care of the correct indentation level. If you place a parameter at the wrong location, it will often be ignored, sometimes even without warning! A common mistake at the beginning is to place model specific parameters outside of the <model_name> scope:
> ```yaml
> parameter_space:
>   SIR:
>     model_parameter: 1   # Parameters in this scope are passed to the model!
> ```

In general, every aspect of running, evaluation, and configuring models is controllable from the configuration file. Take a look at the [documentation entry](https://docs.utopia-project.org/html/ref/mv_base_cfg.html?highlight=worker%20manager#utopya-multiverse-base-configuration) for a full overview of the keys and controls at your disposal. 

### Automatic parameter validation
Take a look at, for example, the `models/SIR/SIR_cfg.yml` file. You will notice lots of little `!is-positive` or `!is-positive-or-zero` flags. These are so-called *validation flags*, and can only be used in the default configuration. They are optional, but their function is to make sure you do not pass invalid parameters to the model (e.g. negative values where only positive ones are allowed), and to catch such errors before the model is run. Running a model with invalid parameters can sometimes lead to cryptic error messages or are even not caught at all, leading to unpredictable behaviour which can be a nightmare to debug. For this reason, you can add these validation flags to the default configuration, along with possible values, ranges, or datatypes for each parameter.

> [!TIP]
> See the [full tutorial entry](https://docs.utopia-project.org/html/usage/run/config-validation.html) for a guide on how to use these. They are useful if you wish to implement your own model.

### Full model configuration
Inside our `utopya_output/SIR` output folder, take a look at the `config` folder. You will see a whole bunch of configuration files. Every single level of the configuration hierarchy is backed up to this folder, allowing you to always reconstruct which parameters you used to run a model. A couple of useful pointers:

- the `model_cfg.yml` file contains the default configuration
- the `run_cfg.yml` is the run configuration
- the `update_cfg.yml` contains any additional parameters you passed from the CLI
- the `meta_cfg.yml` is the combination of all three, plus all the other defaults (many provided by utopya itself) used to run the model. This file will probably seem very large and overwhelming, and you don't really need to worry about it. However, when in doubt, you can refer to it to check where in your custom configuration you need to place certain keys.

> [!TIP]
> Almost every aspect of running, evaluation, and configuring models is controllable from the configuration file. Take a look at the [documentation entry](https://docs.utopia-project.org/html/ref/mv_base_cfg.html?highlight=worker%20manager#utopya-multiverse-base-configuration) for a full overview of the keys and controls at your disposal.

## Evaluation and plotting 

As you saw, calling 
```commandline
utopya run <model_name>
```
performs a series of tasks:

1. It collects all the configuration files, parameters passed, backs up the files, validates parameters, and prepares sweep runs (if configured)
1. It passes the parameters to the model (or models, if running a sweep)
1. It then collects and bundles the output data and stores it
1. Finally, it loads all the data into a so-called `DataManager` and plots the files.

Running a simulation and plotting the data are seperate steps that can be run indepedently of one another. For instance, if you call
```commandline
utopya run <model_name> --no-eval
```
the evaluation step will be skipped. A common use case however will be re-evaluating a model run you have already performed. This can easily be done by running the command
```commandline
utopya eval <model_name>
```
This will re-evaluate the *last* simulation run that was performed. If you wish to evaluate a different run, simply pass the path to that folder in the CLI:
```commandline
utopya eval <model_name> path/to/folder
```
Calling this will use all the plots given in the *default plot configuration file* `<model_name>_plots.yml`. This is the default behaviour; you can pass a different plot configuration using the `--plots-cfg` flat in the CLI:

```commandline
utopya eval <model_name> --plots-cfg path/to/config.yml
```
Take a look at the `SIR_plots.yml` file: you will see a list of entries, each corresponding to one plot. In each of the configuration folders, you will notice an `eval.yml` file. These are plot configurations used for these specific configuration sets; thus, all the configuration set `--cs` flag is is a shorthand for the command

```commandline
utopya run <model_name> path/to/run.yml --plots-cfg path/to/eval.yml
```

Many of these plots are based on a *base plot*: these are default plots given in the `SIR_base_plots.yml` file and which are available throughout the model, i.e. to any other plot configuration. This is handy, since you may wish to share plots throughout the model and not want to have to copy the configuration each time. Take a look at the `SIR_base_plots.yml` file, and scroll all the way down to the `loss` baseplot:

```yaml
loss:
  based_on:
    - .creator.universe
    - .plot.facet_grid.line
  select:
    data: loss
```
This function plots the training loss for each batch, and is available throughout the model. Let's go through it line by line: the `based_on` argument tells the `PlotManager` which configurations to use as the base. Remember that in utopya, a single run is called a `universe`, and that sweeping over multiple parameters creates multiple universes, or `multiverses`. The two plot creators to use are thus the `.creator.universe` and the `.creator.multiverse`. The universe creator creates plots for each individual universe, whereas the multiverse creator creates plots for the multiverse. The `.plot.facet_grid.line` is the plot function to use to plot a line. Finally, the `select` key tells the `PlotManager` which data to plot. It's that simple. Everything else shown in the configuration entry is just styling, which you can also control right from the configuration (and this backed up and reconstructible later on). If you now wish to use this function in your model evaluation, create an `eval.yml` and simply add
```yaml
loss:
  based_on: loss  # This is the 'loss' plot from the base configuration
```

> [!TIP]
> Read the [full tutorial entry](https://docs.utopia-project.org/html/usage/eval/plotting/index.html) on plotting before continuing to the next steps.

The advantage of configuration-based plotting is twofold: for one, it once again means the configuration files are stored alongside plots, meaning any given plot can be quickly recreated, and you will always be able to understand what you did to create a specific plot long after you first made it. This is invaluable for scientific research, where workflows often involve a lot of experimenting and playing around with numerical settings, and you may wish to return to a previous configuration weeks or months later. The other advantage is that utopya supports data transformation right from the configuration file: this means that data analysis and data plotting are kept seperate, and you can always reconstruct the analysis steps later. 

> [!TIP]
> Read the [full tutorial entry](https://docs.utopia-project.org/html/usage/eval/dag/index.html) on configuration-based analysis using a DAG (directed acyclic graph). utopya uses [xarray](https://docs.xarray.dev/en/stable/) for data handling and transformation.

## Adjusting the neural net configuration
### Adjusting the architecture
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

### Setting the activation functions
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
This will initialise each parameter with a separate prior. Take a look at the output folder for the `Predictions_on_smooth_data` run; it contains a plot of the initial value distribution:

<img src="https://github.com/ThGaskin/NeuralABM/files/13854397/initial_values.pdf" width=100%>

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

> [!NOTE]
> Specifying the parameters to learn is not supported in the `HarrisWilsonNW` and `Kuramoto models`, since these learn the entire network adjacency matrix.

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
### Loading data
By default, new synthetic data is produced during every run, but this is often not desired. For one, when performing a multiverse run, we want each universe to calibrate the same data. For another, we will want to be able to load in real data. The specific loading syntax for each model is slightly (unifying this is still WIP), but the general concept is always the same: to your run config, add the following entry (here using SIR as an example):

```yaml
SIR:
  Data:
    load_from_dir: load_from_dir: data/SIR/ABM_data/data/uni0/data.h5
```
This will load in the training data from the given `h5` file and use it across universes. See the model-specific README files to see the syntax for each model. Data is stored in the `data/` folder.

## Models overview
This repository contains the following models:
- [**SIR**](models/SIR/README.md): An SDE model of contagious diseases with scalar parameters that are learned from data.
- [**Kuramoto**](models/Kuramoto/README.md): A linear SDE model of synchronisation of network osciallations. The network adjacency matrix is learned from data.
- [**HarrisWilson**](models/HarrisWilson/README.md): A non-linear SDE model of optimal transport, modelling the flow of supply and demand on a network. Scalar parameters are learned from data. 
- [**HarrisWilsonNW**](models/HarrisWilsonNW/README.md): The Harris-Wilson model, but learning the network adjacency matrix from data. The physical equations
- [**Covid**](models/Covid/README.md): A complex model of contagion and the spread of Covid-19. Scalar parameters are learned from data.
  
See the model-specific README files for a guide to each model. The README files are located in the respective `<model_name>` folders.

## Building your own model
If you are ready to build your own `NeuralABM` model, there is an easy command you can use to get started:
```commandline
utopya models copy <model_name>
```
This command will duplicate an existing model and rename it to whatever name you give when prompted. You can then successively change an existing model to your own requirements.
