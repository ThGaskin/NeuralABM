import time

import h5py as h5
import numpy as np
import torch

""" OLS regression function for the Kuramoto model """


def regression(
    training_data: torch.Tensor,
    eigenfrequencies: torch.Tensor,
    h5file: h5.File,
    dt: float,
    *,
    alpha: float,
    beta: float,
    kappa: float
):
    """Estimates the network from first- or second-order dynamics via an OLS-regression, and stores the predictions in
    an h5.File.

    :param training_data: the training data from which to estimate the network
    :param eigenfrequencies: the eigenfrequencies of the nodes
    :param h5file: the h5 file to write the predictions to
    :param dt: the time differential of the forward Euler method
    :param alpha: the inertia coefficient
    :param beta: the friction coefficient
    :param kappa: the coupling coefficient
    """

    # Extract the number of nodes from the training data
    N = training_data.shape[2]

    if alpha == 0:
        # Stack the observations into a matrix for each node
        X = torch.transpose(
            torch.reshape(
                beta * torch.diff(training_data, dim=1) / dt
                - eigenfrequencies[:, :-1, :, :],
                (-1, N),
            ),
            0,
            1,
        )

        # Stack the sine-couplings into a matrix for each node
        G = torch.zeros(
            training_data.shape[0],
            training_data.shape[1] - 1,
            training_data.shape[2],
            N,
        )

    else:
        velocities = torch.diff(training_data, dim=1) / dt

        X = torch.transpose(
            torch.reshape(
                alpha * torch.diff(velocities, dim=1) / dt
                + beta * velocities[:, :-1, :, :]
                - eigenfrequencies[:, 2:, :, :],
                (-1, N),
            ),
            0,
            1,
        )
        G = torch.zeros(
            training_data.shape[0],
            training_data.shape[1] - 2,
            training_data.shape[2],
            N,
        )

    t_0 = 1 if alpha != 0 else 0
    for dset in range(G.shape[0]):
        for step in range(0, G.shape[1]):
            G[dset, step] = torch.sin(
                -training_data[dset, step + t_0]
                + torch.flatten(training_data[dset, step + t_0])
            )

    # Permute node and time series indices
    G = torch.reshape(torch.permute(G, (2, 3, 0, 1)), (N, N, -1))

    # Create an h5 Group for the regression output
    regression_group = h5file.require_group("regression_data")

    # The predicted matrix, estimated column-wise
    A = torch.zeros(N, N)

    # Transpose G
    G_transpose = torch.transpose(G, 1, 2)

    # Keep track of time
    start_time = time.time()

    # Estimate the coupling vector for each node
    for n in range(N):
        g = torch.cat((G[n][:n, :], G[n][n + 1 :, :]))
        g_t = torch.transpose(g, 0, 1)
        t = torch.matmul(G[n], G_transpose[n])
        subsel = torch.cat((t[:n, :], t[n + 1 :, :]), dim=0)
        subsel = torch.cat((subsel[:, :n], subsel[:, n + 1 :]), dim=1)

        row_entry = np.matmul(torch.matmul(X[n], g_t), np.linalg.inv(subsel))
        A[n, :] = (
            1 / kappa * torch.cat((row_entry[:n], torch.tensor([0]), row_entry[n:]))
        )

    # Write out the time it took to run OLS
    prediction_time = time.time() - start_time

    # Create data group and datasets for the predictions
    dset_prediction = regression_group.create_dataset(
        "predictions",
        (N, N),
        maxshape=(N, N),
        chunks=True,
        compression=3,
    )
    dset_prediction.attrs["dim_names"] = ["i", "j"]
    dset_prediction.attrs["coords_mode__i"] = "trivial"
    dset_prediction.attrs["coords_mode__j"] = "trivial"
    dset_prediction[:, :] = A

    dset_time = regression_group.create_dataset(
        "computation_time",
        (1,),
        maxshape=(1,),
        chunks=True,
        compression=3,
    )
    dset_time.attrs["dim_names"] = ["training_time"]
    dset_time[-1] = prediction_time


def rank(training_data: torch.Tensor, h5file: h5.File, *, alpha: float):
    """Estimates the network from first- or second-order dynamics via an OLS-regression, and stores the predictions in
    an h5.File.
    :param training_data: the training data from which to estimate the network
    :param h5file: the h5 file to write the predictions to
    :param alpha: the coefficient of the second derivative
    """

    # Extract the number of nodes from the training data
    N = training_data.shape[2]

    if alpha == 0:
        # Stack the sine-couplings into a matrix for each node
        G = torch.zeros(
            training_data.shape[0],
            training_data.shape[1] - 1,
            training_data.shape[2],
            N,
        )

    else:
        G = torch.zeros(
            training_data.shape[0],
            training_data.shape[1] - 2,
            training_data.shape[2],
            N,
        )

    t_0 = 1 if alpha != 0 else 0
    for dset in range(G.shape[0]):
        for step in range(0, G.shape[1]):
            G[dset, step] = torch.sin(
                -training_data[dset, step + t_0]
                + torch.flatten(training_data[dset, step + t_0])
            )

    # Permute node and time series indices
    G = torch.reshape(torch.permute(G, (2, 3, 0, 1)), (N, N, -1))

    # Create an h5 Group for the regression output
    regression_group = h5file.require_group("regression_data")

    # Transpose G
    G_transpose = torch.transpose(G, 1, 2)

    ranks = np.zeros(N)

    # Estimate the coupling vector for each node
    for n in range(N):
        t = torch.matmul(G[n], G_transpose[n])
        subsel = torch.cat((t[:n, :], t[n + 1 :, :]), dim=0)
        subsel = torch.cat((subsel[:, :n], subsel[:, n + 1 :]), dim=1)
        ranks[n] = np.linalg.matrix_rank(subsel)

    # Create data group and datasets for the predictions
    dset_rank = regression_group.create_dataset(
        "rank",
        (N,),
        maxshape=(N,),
        chunks=True,
        compression=3,
    )
    dset_rank.attrs["dim_names"] = ["vertex_idx"]
    dset_rank.attrs["coords_mode__vertex_idx"] = "trivial"
    dset_rank[:,] = ranks
