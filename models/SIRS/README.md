Neural parameter inference for the SIRS model of infection
---
This model is a simple toy model of the spread of infection: the Susceptible-Infected-Recovered-Susceptible compartmental
model. Three comparments of susceptible, infected, and recovered agents interact via
$$S \overset{k_I}{\longrightarrow} I \overset{k_R}{\longrightarrow} R \overset{k_S}{\longrightarrow} S.$$
Unlike the standard SIR model, agents can lose their immunity and move back into the 'Susceptible' compartment with 
probability $k_S$. The system of ODEs thus reads:
$$ \begin{gather} \partial_t S = -k_I SI + k_S R \\ \partial_t I = k_I SI - k_R I \\ \partial_t R = k_R I - k_S R \end{gather}$$
By setting $k_S=0$, we recover the conventional SIR model.

### Quickstart
We recommend starting with this model to familiarise yourself with the 
basics of neural parameter estimation. Take a look at the `SIRS_demo.ipynb` Jupyter notebook: it
contains a step-by-step guide to generating data and inferring constant and time-dependent parameters.

### Numerical solvers
We provide a number of numerical solvers for the ODEs in the `model.py` file: a standard Euler method, as well as two
Runge-Kutta solvers (RK4 and Dopri5).

### Ensemble training with `utopya`
For parallelised ensemble training, a basic `utopya` model is provided in the `ensemble_training/run.py` folder. 
This requires having installed `utopya` according to the instructions in the main README, and is computationally a little more advanced and may not be
immediately required for your purposes; however, ensemble training is useful for uncertainty quantification, as well as 
hyperparameter tuning of neural network models.

The basic command
```commandline
utopya run SIRS
```
will run a single model instance and infer two parameters, $k_I$ and $k_R$. The results will be the same as the first 
example given in the Jupyter notebook:

To train a family of neural networks on the same (noisy) time series data, take a look at the `SIR_example`
configuration set: we can run it by calling
```commandline
utopya run SIRS --cs SIR_example
```
This trains a family of 100 neural networks in parallel, and calculates the marginal
densities on both parameters. The training data is located in the `data/SIR_data` files, and 
is plotted alongside the estimates:

You can create or deposit your own files in the `data` folder and load them in for calibration.

### Controlling the `utopya` model from the config
The model is controlled from a configuration file, so you do not need to modify any of the Python code
to change the training or model settings.
