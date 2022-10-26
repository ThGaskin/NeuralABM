# Datasets for the SIR agent-based model of infectious diseases

This folder contains synthetic data of the SIR ABM that can be used to run the data.
Simply set the `load_from_dir` entry in the config to the folder containing the data, for
example:

```yaml
SIR:
  Data:
    load_from_dir: data/SIR/ABM_data_2/data/uni0/data.h5
```

The data will be automatically loaded and used to train the model.
