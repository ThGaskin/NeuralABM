import sys
from os.path import dirname as up
import os
import pathlib
import shutil

import numpy as np
import xarray as xr
from dantro._import_tools import import_module_from_path
from pkg_resources import resource_filename
from dantro.data_mngr import DataManager
from dantro.containers import PassthroughContainer
from dantro.plot.funcs.generic import facet_grid
from dantro.plot import PyPlotCreator
from utopya.yaml import load_yml
from dantro.tools import DoNothingContext
import matplotlib.pyplot as plt
import logging
import pytest

sys.path.insert(0, up(up(up(__file__))))

plot = import_module_from_path(
    mod_path=up(up(up(__file__))), mod_str="model_plots.prob_density"
)

CFG_FILENAME = resource_filename("tests", "cfgs/prob_density.yml")
test_cfg = load_yml(CFG_FILENAME)

from . import (
    ABBREVIATE_TEST_OUTPUT_DIR,
    TEST_OUTPUT_DIR,
    USE_TEST_OUTPUT_DIR,
)

# Disable matplotlib logger (much too verbose)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def create_nd_data(
        n: int, *, shape=None, **dset_kwargs
) -> xr.Dataset:
    """Creates n-dimensional random dataset of a certain shape, containing x, y, and yerr variables. If no shape is
    given, will use ``(2, 3, 4, ..)``. Coordinates are also added to the data. Default length for x, y, and yerr is 20.
    """
    if shape is None:
        shape = tuple([20, *range(2, n + 2)])

    coords = dict()
    for i, d in enumerate(shape):
        if i == 0:
            coords["coords_x"] = np.arange(d)
        else:
            coords[f"dim_{i}"] = np.arange(d)

    return xr.Dataset(
        data_vars=dict(x=(list(coords.keys()), np.random.random(shape)),
                       y=(list(coords.keys()), np.random.random(shape)),
                       yerr=(list(coords.keys()), np.random.random(shape))),
        coords=coords, **dset_kwargs
    )


@pytest.fixture
def tmpdir_or_local_dir(tmpdir, request) -> pathlib.Path:
    """If ``USE_TEST_OUTPUT_DIR`` is False, returns a temporary directory;
    otherwise a test-specific local directory within ``TEST_OUTPUT_DIR`` is
    returned.
    """
    if not USE_TEST_OUTPUT_DIR:
        return tmpdir

    if not ABBREVIATE_TEST_OUTPUT_DIR:
        # include the module and don't do any string replacements
        test_dir = os.path.join(
            TEST_OUTPUT_DIR,
            request.node.module.__name__,
            request.node.originalname,
        )
    else:
        # generate a shorter version without the module and with the test
        # prefixes dropped
        test_dir = os.path.join(
            TEST_OUTPUT_DIR,
            request.node.originalname.replace("test_", ""),
        )

    # Clean out that directory and then recreate it
    print(f"Using local test output directory:\n  {test_dir}")
    if os.path.isdir(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir, exist_ok=True)

    return pathlib.Path(test_dir)


@pytest.fixture
def dm(tmpdir_or_local_dir):
    """Returns a data manager populated with some high-dimensional test data"""
    # Add xr.Datasets for testing

    # Initialize it to a temporary directory and without load config
    _dm = DataManager(tmpdir_or_local_dir)

    grp_dataset = _dm.new_group("datasets")

    # Add Datasets, going up to 4 dimensions
    grp_dataset.add(
        *[
            PassthroughContainer(name=f"{n + 1}D", data=create_nd_data(n))
            for n in range(5)
        ]
    )

    return _dm


def test_prob_density(dm, tmpdir_or_local_dir):
    """ Tests the functionality of the prob_density plot function """

    # The plot manager to test everything with
    ppc = PyPlotCreator("test_prob_density", dm=dm, plot_func=facet_grid)
    ppc._exist_ok = True

    out_path = lambda name: dict(out_path=os.path.join(tmpdir_or_local_dir, name + ".pdf"))

    # Make sure there are no figures currently open, in order to be able to
    # track whether any figures leak from the plot function ...
    plt.close("all")
    assert len(plt.get_fignums()) == 0

    # Invoke the plotting function with data of different dimensionality
    for case_name, config in test_cfg.items():
        if case_name.startswith('_'):
            continue
        context = DoNothingContext()
        with context:
            ppc(
                **out_path(
                    "{case:}".format(
                        case=case_name,
                    )
                ),
                kind='density',
                **config
            )

    # The last figure should survive from this.
    assert len(plt.get_fignums()) == 1
