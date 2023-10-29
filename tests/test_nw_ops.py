import sys
from os.path import dirname as up

import numpy as np
import xarray as xr
from dantro._import_tools import import_module_from_path
from pkg_resources import resource_filename

from utopya.yaml import load_yml

sys.path.insert(0, up(up(up(__file__))))

ops = import_module_from_path(
    mod_path=up(up(up(__file__))), mod_str="model_plots.data_ops"
)

nw_ops = import_module_from_path(
    mod_path=up(up(up(__file__))), mod_str="model_plots.nw_ops"
)

# Load the test config
CFG_FILENAME = resource_filename("tests", "cfgs/data_ops.yml")
test_cfg = load_yml(CFG_FILENAME)


# ----------------------------------------------------------------------------------------------------------------------
# ADJACENCY MATRIX OPERATIONS
# ----------------------------------------------------------------------------------------------------------------------
def test_triangles():
    """Tests the calculation of triangles on adjacency matrices."""

    # Generate some samples of adjacency matrices
    N_samples, N_vertices = 10, 10
    A_samples = np.ones((N_samples, N_vertices, N_vertices))
    for i in range(len(A_samples)):
        np.fill_diagonal(A_samples[i], 0)

    A_samples = xr.DataArray(
        A_samples,
        dims=["samples", "i", "j"],
        coords=dict(
            samples=np.arange(N_samples),
            i=np.arange(N_vertices),
            j=np.arange(N_vertices),
        ),
    )

    triangles = nw_ops.triangles(A_samples, input_core_dims=["j"])
    assert set(triangles.coords.keys()) == {"samples", "i"}

    # Assert number of triangles on each node for complete graph is equal to (N_vertices-1)*(N_vertices-2)
    # (counting directed triangles)
    assert all(triangles.data.flatten() == (N_vertices - 1) * (N_vertices - 2))

    # Bin the triangles along a dimension
    triangles_binned = ops.hist(
        triangles, dim="i", bins=100, ranges=[0, 100], use_bins_as_coords=True
    )

    # Assert that all triangles are in one bin, namely the one at (N_vertices-1)*(N_vertices-2)
    assert all(
        triangles_binned.sel(
            {"x": (N_vertices - 1) * (N_vertices - 2)}, method="nearest"
        ).data
        == N_vertices
    )
    assert triangles_binned.equals(
        nw_ops.binned_nw_statistic(
            triangles, bins=100, ranges=[0, 100], sample_dim="samples"
        )
    )


def test_sel_matrix_indices():
    """Tests selecting matrix indices"""
    N_samples, N_vertices = 10, 10
    n_indices = 5
    A_samples = xr.DataArray(
        np.random.rand(N_samples, N_vertices, N_vertices),
        dims=["sample", "i", "j"],
        coords=dict(
            sample=np.arange(N_samples),
            i=np.arange(N_vertices),
            j=np.arange(N_vertices),
        ),
        name="_A",
    )

    indices = xr.Dataset(
        data_vars=dict(
            i=(["idx"], np.random.randint(0, N_vertices, n_indices)),
            j=(["idx"], np.random.randint(0, N_vertices, n_indices)),
        ),
        coords=dict(idx=np.arange(n_indices)),
    ).expand_dims(dict(sample=np.arange(N_samples)))

    selection = nw_ops.sel_matrix_indices(A_samples, indices, exclude_dim=["sample"])
    assert set(selection.coords) == {"idx", "sample", "i", "j"}
    assert len(selection.coords["idx"]) == n_indices

    selection = nw_ops.sel_matrix_indices(
        A_samples, indices, exclude_dim=["sample"], drop=True
    )
    assert set(selection.coords) == {"idx", "sample"}


def test_largest_entry_indices():
    """Tests returning the indices of the largest entries"""
    N_vertices = 10
    A = xr.DataArray(
        [[i + j for j in range(N_vertices)] for i in range(N_vertices)],
        dims=["i", "j"],
        coords=dict(i=np.arange(N_vertices), j=np.arange(N_vertices)),
    )

    # Get the indices for the symmetric case
    indices = nw_ops.largest_entry_indices(A, 4, symmetric=False)
    assert set(indices.data_vars) == {"i", "j", "value"}
    assert set(indices.coords.keys()) == {"idx"}

    assert indices.isel({"idx": 0})["i"] == N_vertices - 1
    assert indices.isel({"idx": 0})["j"] == N_vertices - 1
    assert indices.isel({"idx": 0})["value"] == 2 * (N_vertices - 1)

    # Repeat for asymmetric case
    indices = nw_ops.largest_entry_indices(A, 4, symmetric=True)
    assert indices.isel({"idx": -1})["i"] == N_vertices - 3
    assert indices.isel({"idx": -1})["value"] == 16

    # Select the entries from A and check they are equal to values obtained
    assert all(
        indices["value"].data == nw_ops.sel_matrix_indices(A, indices[["i", "j"]]).data
    )


# ----------------------------------------------------------------------------------------------------------------------
# DISTRIBUTION OPERATIONS
# ----------------------------------------------------------------------------------------------------------------------

# TODO: Add tests
