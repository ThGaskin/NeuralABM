import sys
from os.path import dirname as up

import h5py as h5
import pytest
import torch
from dantro._import_tools import import_module_from_path
from pkg_resources import resource_filename

from utopya.yaml import load_yml

sys.path.insert(0, up(up(up(__file__))))

SIR = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="models.SIR")

# Load the test config
CFG_FILENAME = resource_filename("tests", "cfgs/SIR_DataGeneration.yml")
test_cfg = load_yml(CFG_FILENAME)


# Test that ABM data and smooth data are generated
def test_data_generation(tmpdir):
    # Create an h5File in the temporary directory for the
    h5dir = tmpdir.mkdir("hdf5_data")

    for name, config in test_cfg.items():
        h5file = h5.File(h5dir.join(f"test_{name}.h5"), "w")
        h5group = h5file.create_group("SIR")

        synthetic_data = SIR.get_SIR_data(
            data_cfg=config, h5group=h5group, write_init_state=False
        )

        n = config["synthetic_data"]["num_steps"]
        assert len(synthetic_data) == n

        # Check the densities are consistent in the noiseless case, and non-negative in the noisy case
        sigma = config["synthetic_data"].pop("sigma", 0)
        if sigma == 0:
            assert (
                torch.round(torch.sum(synthetic_data, dim=1), decimals=4)
                == torch.tensor([1.0])
            ).all()
        else:
            assert (synthetic_data >= 0).all()

        # Check infection has taken place
        assert torch.max(synthetic_data, axis=0)[1][1] > synthetic_data[-1][1]

        # Check recovery has taken place
        assert synthetic_data[0][0] > synthetic_data[-1][0]
        assert synthetic_data[-1][-1] > synthetic_data[0][-1]

        # Check data has been written to the h5Group
        if config["synthetic_data"]["type"] == "from_ABM":
            assert list(h5group.keys()) == ["kinds", "position", "true_counts"]
        elif config["synthetic_data"]["type"] == "smooth":
            assert list(h5group.keys()) == ["true_counts"]
