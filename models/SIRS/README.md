Neural parameter inference for the SIRS model of infection
---
The Susceptible-Infected-Recovered-Susceptible compartmental
model is a simple toy model of the spread of infection. Three comparments of susceptible, infected, and recovered agents
interact via the following reaction scheme:

$$S \overset{k_I}{\longrightarrow} I \overset{k_R}{\longrightarrow} R \overset{k_S}{\longrightarrow} S.$$

Unlike the standard SIR model, agents can lose their immunity and move back into the 'Susceptible' compartment with
probability $k_S$. The system of ODEs thus reads:

$$ \begin{gather} \partial_t S = -k_I SI + k_S R \\
\partial_t I = k_I SI - k_R I \\
\partial_t R = k_R I - k_S R
\end{gather}$$

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
This requires having installed `utopya` according to the instructions in the main README, and is computationally a
little more advanced and may not be
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

## Hybrid modelling with the SIRS model
In the `SIRS_hybrid_experiments.ipynb` Jupyter notebook we experiment with various hybridisation variants of the SIRS model, investigating how the degree of
structural knowledge of the model influences the learning performance. All the relevant code is 
supplied in the `hybrid_models` submodule in this folder. We investigate four levels of hybridisation:

**Parameters only**

In this model, the only thing to be inferred are the constant parameters $(k_S, k_I, k_R)$. We parametrise these as a
neural network $u_\theta$, using either the observed data $\mathbf{Y} = (S(t), I(t), R(t))$ as input, or the
self-generated estimated solution to the ODE (`recursive').

**Time-varying component/Hybrid-1**

Here, we infer two constant parameters, $k_R$ and $k_S$, as well as the time-varying component $k_I S$. The equations
thus become

$$\begin{align}
\mathrm{d}S & = \lambda_2 R - \lambda_1(t) I \\
\mathrm{d}I & = \lambda_1(t) I - \lambda_3 I \\
\mathrm{d} R &= \lambda_3 I - \lambda_2 R
\end{align}$$

with $\lambda_1$ the output of a neural network $u_{\theta_1}(\mathbf{x}(t))$, and the constant parameters the
output of a different network $u_{\theta_2}(\mathbf{Y})$. The first takes a single, time-dependent state (observed
or self-predicted) as input, the second takes an entire time series as input.

**Unidentifiable formulation/Hybrid-2**

Here we identify three time-dependent components:

$$\begin{align}
\mathrm{d}S & = -\lambda_1 + \lambda_3 \\
\mathrm{d}I & = \lambda_1 - \lambda_2 \\
\mathrm{d} R &= \lambda_2 - \lambda_3
\end{align}$$

This formulation is not identifiable. A single neural network is used to map the current state $\mathbf{x}(t)$ to
the missing component vector $\mathbf{\lambda}(t) = (\lambda_1, \lambda_2, \lambda_3)$.

**Neural ODE**
Here, the entire right-hand side of the equation is replaced by a neural network:

$$\begin{equation}
\begin{pmatrix}\mathrm{d}S \\ \mathrm{d}I \\ \mathrm{d}R \end{pmatrix} = u_\theta
\end{equation}$$

Here again, the input to the neural network can either be the observed or self-generated data.
