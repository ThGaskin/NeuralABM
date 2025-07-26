# Datasets for the Harris-Wilson model of economic activity

This folder contains several datasets, both real and synthetic, that can be used to train the neural net and learn
parameters for the Harris-Wilson equations. Simply set the `load_from_dir` key in the `run_cfg.yml` file to point
to the folder containing the data: the data will automatically be loaded and
the model trained on that data. When plotting, take care to adjust any `true_parameters` keys if you want
to compare the model predictions to the ground truths.

## Synthetic data
The `synthetic_data` folder contains synthetically generated networks, origin sizes, and destination sizes. The name of the
folder indicates the network size, i.e. `N_100_M_10` means N=100 and M=10.
The specific configurations for each dataset are given by the `config.yml` files in the folders.

## London data
The `London_data` folder contains datasets of economic activity across Greater London. The `GLA_data` folder
contains the data compiled from the two GLA studies on ward profiles and
retail floor space. The `dest_sizes.csv` and `origin_sizes.csv` are the destination and
origin zone sizes used in the paper. The `exp_times.csv` and `exp_distances.csv`
are the two different transport network metrics used, calculated via
`exp(-d_{ij}/max(d_{ij}))` from the respective `distances.csv` and `times.csv` files. The `Google_Distance_Matrix_Data` folder contains transport times and distances using
the Google Maps API service. Each file is a pkl-dictionary containing the API output for different
travel modes: `transit` (public transport) and `driving` (driving, no traffic).
The `departure_time` for transit is Sunday, June 19th 2022, 1 pm GMT (in Unix time: `departure_time = 1655640000`). However, since
trips in the past cannot be computed, a future date must always be specified when using the
API. The data is also available as a cost matrix in `.csv` format: entries are given in seconds and metres
respectively.

Also available are Euclidean distances, and the densities from running the MCMC analysis at
[this repository](https://github.com/lellam/cities_and_regions).

**Data sources:**
- Greater London Authority (GLA), [Ward Profiles and Atlas](https://data.london.gov.uk/dataset/ward-profiles-and-atlas) 2016.
- GLA, [London Town Centre Health Check Analysis Report](https://data.gov.uk/dataset/2a50ca67-954a-4f22-91d8-d3dfe9116143/london-town-centre-health-check-analysis-report) 2018.
- Google Maps [Distance Matrix API](https://developers.google.com/maps/documentation/distance-matrix/overview).
- GLA, [Statistical GIS Boundary Files for London](https://data.london.gov.uk/dataset/statistical-gis-boundary-files-london), 2011.
