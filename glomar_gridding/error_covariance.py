# Copyright 2025 National Oceanography Centre
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Functions for computing correlated and uncorrelated components of the error
covariance. These values are determined from standard deviation (sigma) values
assigned to groupings within the observational data.

The correlated components will form a matrix that is permutationally equivalent
to a block diagonal matrix (i.e. the matrix will be block diagonal if the
observational data is sorted by the group).

The uncorrelated components will form a diagonal matrix.

Further a distance-based component can be constructed, where distances between
records within the same grid box are evaluated.

The functions in this module are valid for observational data where there could
be more than 1 observation in a gridbox.
"""

from collections.abc import Callable
from warnings import warn

import numpy as np
import polars as pl

from glomar_gridding.distances import haversine_distance_from_frame

from .utils import ColumnNotFoundError, check_cols


def uncorrelated_components(
    df: pl.DataFrame,
    group_col: str = "data_type",
    obs_sig_col: str | None = None,
    obs_sig_map: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Calculates the covariance matrix of the measurements (observations). This
    is the uncorrelated component of the covariance.

    The result is a diagonal matrix. The diagonal is formed by the square of the
    sigma values associated with the values in the grouping.

    The values can either be pre-defined in the observational dataframe, and
    can be indicated by the "bias_val_col" argument. Alternatively, a mapping
    can be passed, the values will be then assigned by this mapping of group to
    sigma.

    Parameters
    ----------
    df : polars.DataFrame
        The observational DataFrame containing values to group by.
    group_col : str
        Name of the group column to use to set observational sigma values.
    obs_sig_col : str | None
        Name of the column containing observational sigma values. If set and
        present in the DataFrame, then this column is used as the diagonal of
        the returned covariance matrix.
    obs_sig_map : dict[str, float] | None
        Mapping between group and observational sigma values used to define
        the diagonal of the returned covariance matrix.

    Returns
    -------
    A diagonal matrix representing the uncorrelated components of the error
    covariance matrix.
    """
    if obs_sig_col is not None and obs_sig_col in df.columns:
        return np.diag(df.get_column(obs_sig_col))
    elif obs_sig_col is not None and obs_sig_col not in df.columns:
        raise ColumnNotFoundError(
            f"Observation Bias Column {obs_sig_col} not found."
        )

    obs_sig_map = obs_sig_map or {}
    groupings: pl.Series = df.get_column(group_col)
    s = groupings.replace_strict(
        {k: v**2 for k, v in obs_sig_map.items()}, default=0.0
    )
    if s.eq(0.0).all():
        warn("No values in obs_covariance set")
    elif s.eq(0.0).any():
        warn("Some values in obs_covariance not set")

    return np.diag(s)


def correlated_components(
    df: pl.DataFrame,
    group_col: str,
    bias_sig_col: str | None = None,
    bias_sig_map: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Returns measurements covariance matrix updated by adding bias uncertainty to
    the measurements based on a grouping within the observational data.

    The result is equivalent to a block diagonal matrix via permutation. If the
    input observational data is sorted by the group column then the resulting
    matrix is block diagonal, where the blocks are the size of each grouping.
    The values in each block are the square of the sigma value associated with
    the grouping.

    Note that in most cases the output is not a block-diagonal, as the input
    is not usually sorted by the group column. In most processing cases, the
    input dataframe will be sorted by the gridbox index.

    The values can either be pre-defined in the observational dataframe, and
    can be indicated by the "bias_val_col" argument. Alternatively, a mapping
    can be passed, the values will be then assigned by this mapping of group to
    sigma.

    Parameters
    ----------
    df : polars.DataFrame
        Observational DataFrame including group information and bias uncertainty
        values for each grouping. It is assumed that a single bias uncertainty
        value applies to the whole group, and is applied as cross terms in the
        covariance matrix (plus to the diagonal).
    group_col : str
        Name of the column that can be used to partition the observational
        DataFrame.
    bias_sig_col : str | None
        Name of the column containing bias uncertainty values for each of
        the groups identified by 'group_col'. It is assumed that a single bias
        uncertainty value applies to the whole group, and is applied as cross
        terms in the covariance matrix (plus to the diagonal).
    bias_sig_map : dict[str, float] | None
        Mapping between values in the group_col and bias uncertainty values,
        if bias_val_col is not in the DataFrame.

    Returns
    -------
    The correlated components of the error covariance.
    """
    check_cols(df, [group_col])

    # Initialise array
    covx = np.zeros((len(df), len(df)))

    bias_sig_col = bias_sig_col or "_bias_uncert"
    bias_sig_map = bias_sig_map or {}

    if bias_sig_col not in df.columns:
        df = df.with_columns(
            pl.col(group_col)
            .replace_strict(
                {k: v**2 for k, v in bias_sig_map.items()},
                default=0.0,
            )
            .alias(bias_sig_col)
        )
        if df[bias_sig_col].eq(0.0).all():
            warn("No bias uncertainty values set")
        elif df[bias_sig_col].eq(0.0).any():
            warn("Some bias uncertainty values not set")

    # NOTE: polars is easier for this analysis!
    df = (
        df.select(group_col, bias_sig_col)
        .with_row_index("index")
        .group_by(group_col)
        # NOTE: It is expected that the bias value should be the same for all
        #       records within the same group
        .agg(pl.col("index"), pl.col(bias_sig_col).first())
    )
    for row in df.rows(named=True):
        if row[bias_sig_col] is None:
            print(f"Group {row[group_col]} has no bias uncertainty value set")
            continue
        # INFO: Adding cross-terms to covariance
        inds = np.ix_(row["index"], row["index"])
        covx[inds] = covx[inds] + row[bias_sig_col]

    return covx


def dist_weight(
    df: pl.DataFrame,
    dist_fn: Callable,
    grid_idx: str = "grid_idx",
    **dist_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the distance and weight matrices over gridboxes for an input Frame.

    This function acts as a wrapper for a distance function, allowing for
    computation of the distances between positions in the same gridbox using any
    distance metric.

    The weightings from this function are for the gridbox mean of the
    observations within a gridbox.

    Parameters
    ----------
    df : polars.DataFrame
        The observation DataFrame, containing the columns required for
        computation of the distance matrix. Contains the "grid_idx" column which
        indicates the gridbox for a given observation. The index of the
        DataFrame should match the index ordering for the output distance
        matrix/weights.
    dist_fn : Callable
        The function used to compute a distance matrix for all points in a given
        grid-cell. Takes as input a polars.DataFrame as first argument. Any
        other arguments should be constant over all gridboxes, or can be a
        look-up table that can use values in the DataFrame to specify values
        specific to a gridbox. The function should return a numpy matrix, which
        is the distance matrix for the gridbox only. This wrapper function will
        correctly apply this matrix to the larger distance matrix using the
        index from the DataFrame.

        If dist_fn is None, then no distances are computed and None is returned
        for the dist value.
    grid_idx : str
        Name of the column containing the grid index values
    **dist_kwargs
        Arguments to be passed to dist_fn. In general these should be constant
        across all gridboxes. It is possible to pass a look-up table that
        contains pre-computed values that are gridbox specific, if the keys can
        be matched to a column in df.

    Returns
    -------
    dist : numpy.matrix
        The distance matrix, which contains the same number of rows and columns
        as rows in the input DataFrame df. The values in the matrix are 0 if the
        indices of the row/column are for observations from different gridboxes,
        and non-zero if the row/column indices fall within the same gridbox.
        Consequently, with appropriate re-arrangement of rows and columns this
        matrix can be transformed into a block-diagonal matrix. If the DataFrame
        input is pre-sorted by the gridbox column, then the result is a
        block-diagonal matrix.

        If dist_fn is None, then this value will be None.
    weights : numpy.matrix
        A matrix of weights. This has dimensions n x p where n is the number of
        unique gridboxes and p is the number of observations (the number of rows
        in df). The values are 0 if the row and column do not correspond to the
        same gridbox and equal to the inverse of the number of observations in a
        gridbox if the row and column indices fall within the same gridbox. The
        rows of weights are in a sorted order of the gridbox. Should this be
        incorrect, one should re-arrange the rows after calling this function.
    """
    # NOTE: Sorting the unique grid-cell values ensures consistency between runs
    #       if the frame ordering gets shuffled in some way.
    # NOTE: Alternatively sort by "flattened_idx"
    gridboxes = sorted(df[grid_idx].unique())
    _n_gridboxes = len(gridboxes)
    _n_obs = df.height

    df = df.with_row_index("_index")

    # Initialise
    weights = np.zeros((_n_gridboxes, _n_obs))
    dist = np.zeros((_n_obs, _n_obs))

    for i, gridbox_df in enumerate(df.partition_by(grid_idx)):
        gridbox_idcs = gridbox_df.get_column("_index").to_list()
        idcs_array = np.ix_(gridbox_idcs, gridbox_idcs)

        weights[i, gridbox_idcs] = 1 / gridbox_df.height
        dist[idcs_array] = dist_fn(gridbox_df, **dist_kwargs)

    return dist, weights


def get_weights(
    df: pl.DataFrame,
    grid_idx: str = "grid_idx",
) -> np.ndarray:
    """
    Get just the weight matrices over gridboxes for an input Frame.

    The weightings from this function are for the gridbox mean of the
    observations within a gridbox.

    Parameters
    ----------
    df : polars.DataFrame
        The observation DataFrame, containing the columns required for
        computation of the distance matrix. Contains the "grid_idx" column which
        indicates the gridbox for a given observation. The index of the
        DataFrame should match the index ordering for the output weights.
    grid_idx : str
        Name of the column containing the gridbox index from the output grid.

    Returns
    -------
    weights : numpy.matrix
        A matrix of weights. This has dimensions n x p where n is the number of
        unique gridboxes and p is the number of observations (the number of rows
        in df). The values are 0 if the row and column do not correspond to the
        same gridbox and equal to the inverse of the number of observations in a
        gridbox if the row and column indices fall within the same gridbox. The
        rows of weights are in a sorted order of the gridbox. Should this be
        incorrect, one should re-arrange the rows after calling this function.
    """
    weights = (
        df.with_row_index("_index")
        .with_columns((1 / pl.len().over(grid_idx)).alias("_weight"))
        .select(["_index", grid_idx, "_weight"])
        .pivot(on=grid_idx, index="_index", values="_weight")
        .fill_null(0)
        .sort("_index")
        .drop("_index")
    )
    return (
        weights.select(sorted(weights.columns, key=int)).to_numpy().transpose()
    )


def _vgm_model(
    psill: float | np.ndarray,
    space_range: float | np.ndarray,
    time_range: float | np.ndarray,
    space_dist: np.ndarray,
    time_dist: np.ndarray,
) -> np.ndarray:
    """
    Apply the spatio-temporal exponential variogram model.

    Compute the expected spatiotemporal covariance between observations
    given their haversine distance and time delta from each other

    See Chapter 2.5 in Cornes et al 2020.

    Parameters
    ----------
    psill : float | numpy.ndarray
        Sill of the variogram where it will flatten out. This value is the
        variance.
    space_range : float | numpy.ndarray
        Spatial length scale [km]
    time_range : float | numpy.ndarray
        Temporal scale [days]
    space_dist : numpy.ndarray
        Distance delta between observations [km]
    time_dist : numpy.ndarray
        Time delta between observations [us]

    Returns
    -------
    numpy.ndarray
        The fitted variogram model
    """
    tau_space = space_dist / space_range
    if np.all(np.logical_not(np.isnat(time_dist))):
        tau_time = (time_dist / np.timedelta64(1, "D")) / time_range
    elif np.all(np.isnat(time_dist)):
        tau_time = time_dist / time_range
    else:
        raise TypeError(
            "Input time_dist must be of inner dtype 'timedelta' or "
            + f"'numpy.timedelta64'. Got {time_dist.dtype = }."
        )
    return psill * np.exp(-(np.square(tau_space) + np.square(tau_time)))


def weighted_sum(
    df: pl.DataFrame,
    grid_idx: str,
    group_col: str,
    error_group_correlated: str,
    error_uncorrelated: str,
    lat_col: str = "lat",
    lon_col: str = "lon",
    date_col: str = "datetime",
    sill: str = "sill",
    space_range: str = "space_range",
    time_range: str = "time_range",
    bad_groups: str | list[str] | None = None,
) -> np.ndarray:
    """
    Get the weights matrix for the weighted sum approach, accounting for
    correlated and uncorrelated error components. The weights are computed using
    uncertainty from a simple spatio-temporal exponential variogram model, the
    correlated error covariance components (group-wise) and the uncorrelated
    error covariance components (group-wise).

    The correlated components of the error covariance are 0 between pairs of
    records with _different_ groups, and the value from the
    'error_group_correlated' column squared when the groups match.

    This computes the weights for each grid-box using the local inverse error
    covariance.

    A factor of 1/4 can be applied to the cross-record pairs in the computation
    of correlated components for a subset of groupings considered "bad" - for
    example generic ids.

    Parameters
    ----------
    df : polars.DataFrame
        The observation DataFrame. Contains the "grid_idx" column which
        indicates the gridbox for a given observation. The index of the
        DataFrame should match the index ordering for the output weights.
    grid_idx : str
        Name of the column containing the gridbox index from the output grid.
    group_col : str
        Name of the column by which records are grouped for the purpose of
        correlated and uncorrelated components of the error covariance
        structure.
    error_group_correlated : str
        Name of the column containing correlated error values by group.
    error_uncorrelated : str
        Name of the column containing uncorrelated error values by group.
    lat_col : str
        Name of the latitude column.
    lon_col : str
        Name of the longitude column.
    date_col : str
        Name of the datetime column.
    bad_groups : str
        Values in the groups that required lower priority in the weightings.
        For example, this could be records with invalid, or generic, ids.
    sill : str
        The name of the column containing sill values of the simplified varigram
        model.
    space_range : str
        The name of the column containing space range values for the simplified
        varigram model. [km]
    time_range : str
        The name of the column containing time range values for the simplified
        varigram model. [days]

    Returns
    -------
    weights : np.ndarray
        A matrix containing the weights for each record in the data.
        Retaining the order within the frame.
    """
    required_cols = [
        grid_idx,
        group_col,
        lat_col,
        lon_col,
        date_col,
        error_group_correlated,
        error_uncorrelated,
        sill,
        space_range,
        time_range,
    ]
    check_cols(df, required_cols)

    bad_groups = bad_groups or []
    n_obs: int = df.height
    n_gridboxes: int = df.get_column(grid_idx).unique().len()
    weights: np.ndarray = np.zeros((n_gridboxes, n_obs), dtype=np.float32)

    df = df.with_columns(pl.col(group_col).fill_null(""))

    df = df.with_row_index("_index")

    sp = df.partition_by(grid_idx, include_key=True)
    sp.sort(key=lambda x: x[grid_idx][0])

    for grid_box, grid_box_frame in enumerate(sp):
        grid_box_idx = grid_box_frame.get_column("_index").to_numpy()
        if grid_box_frame.height == 1:
            weights[grid_box, grid_box_idx[0]] = 1
            continue
        grid_box_weights, *_ = grid_box_weighted_sum(
            grid_box_frame,
            group_col=group_col,
            error_group_correlated=error_group_correlated,
            error_uncorrelated=error_uncorrelated,
            bad_groups=bad_groups,
            lat_col=lat_col,
            lon_col=lon_col,
            date_col=date_col,
            sill=sill,
            space_range=space_range,
            time_range=time_range,
        )
        weights[grid_box, grid_box_idx] = grid_box_weights
    return weights


def grid_box_weighted_sum(
    grid_box_frame: pl.DataFrame,
    group_col: str,
    error_group_correlated: str,
    error_uncorrelated: str,
    lat_col: str = "lat",
    lon_col: str = "lon",
    date_col: str = "datetime",
    sill: str = "sill",
    space_range: str = "space_range",
    time_range: str = "time_range",
    bad_groups: str | list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the weighted mean weights for a grid-box. The weights are computed using
    uncertainty from a simple spatio-temporal exponential variogram model, the
    correlated error covariance components (group-wise) and the uncorrelated
    error covariance components (group-wise).

    The correlated components of the error covariance are 0 between pairs of
    records with _different_ groups, and the value from the
    'error_group_correlated' column squared when the groups match.

    This computes the weights for a grid-box using the local inverse error
    covariance.

    A factor of 1/4 can be applied to the cross-record pairs in the computation
    of correlated components for a subset of groupings considered "bad" - for
    example generic ids.

    Parameters
    ----------
    grid_box_frame : polars.DataFrame
        DataFrame containing the observations for all records within a grid-box.
    group_col : str
        Name of the column by which records are grouped for the purpose of
        correlated and uncorrelated components of the error covariance
        structure.
    error_group_correlated : str
        Name of the column containing correlated error values by group.
    error_uncorrelated : str
        Name of the column containing uncorrelated error values by group.
    lat_col : str
        Name of the latitude column.
    lon_col : str
        Name of the longitude column.
    date_col : str
        Name of the datetime column.
    sill : str
        The name of the column containing sill values of the simplified varigram
        model.
    space_range : str
        The name of the column containing space range values for the simplified
        varigram model. [km]
    time_range : str
        The name of the column containing time range values for the simplified
        varigram model. [days]
    bad_groups : str | list[str] | None
        Values in the groups that required lower priority in the weightings.
        For example, this could be records with invalid, or generic, ids.

    Returns
    -------
    weights : np.ndarray
        A vector containing the weights for each record in the grid-box.
        Retaining the order within the grid-cell.
    vario : np.ndarray
        A matrix containing the output of the spatio-temporal variogram model
        for the grid-box records.
    correlated : np.ndarray
        Correlated error covariance matrix for the grid-box. Contains correlated
        error values when the group between each pair of records matches, zero
        otherwise.
    uncorrelated : np.ndarray
        Uncorrelated error matrx for the grid-box. Diagonal matrix.
    """
    n_obs = grid_box_frame.height
    dist = haversine_distance_from_frame(
        grid_box_frame.select(
            pl.col(lat_col).alias("lat"),
            pl.col(lon_col).alias("lon"),
        )
    )
    dates = grid_box_frame.get_column(date_col).to_numpy()
    dt_diff = np.abs(np.subtract.outer(dates, dates))  # np.timedelta us

    vario = covar_mat = _vgm_model(
        psill=grid_box_frame.get_column(sill).to_numpy(),
        space_range=grid_box_frame.get_column(space_range).to_numpy(),
        time_range=grid_box_frame.get_column(time_range).to_numpy(),
        space_dist=dist,
        time_dist=dt_diff,
    )
    grid_box_groups = grid_box_frame.get_column(group_col).to_numpy()
    rho_beta = np.equal.outer(grid_box_groups, grid_box_groups).astype(
        np.float32
    )
    if bad_groups is not None:
        # Adjustments for records in groups to lower weightings
        # (e.g. generic ids)
        bad_group_idx = np.isin(grid_box_groups, bad_groups)
        rho_beta[:, bad_group_idx] *= 0.25
        rho_beta[bad_group_idx, bad_group_idx] *= 4

    corr_error = grid_box_frame.get_column(error_group_correlated).to_numpy()
    correlated = rho_beta * np.multiply.outer(corr_error, corr_error)

    uncorrelated = np.diag(grid_box_frame.get_column(error_uncorrelated).pow(2))

    covar_mat = vario + correlated + uncorrelated

    ones_n = np.ones((1, n_obs), dtype=covar_mat.dtype)
    zero = np.zeros((1, 1), dtype=covar_mat.dtype)

    covar_mat = np.block([[covar_mat, ones_n.T], [ones_n, zero]])

    rhs = np.append(np.zeros(n_obs), 1)
    weights = np.linalg.solve(covar_mat, rhs)[:-1]

    # Apply correction to adjust for negative weights
    # following Journel and Rao (1996), Yamamoto (2000)
    cmin = min(weights)
    if cmin < 0:
        cmin = abs(cmin)
        weights = (weights + cmin) / sum(weights + cmin)

    return weights, vario, correlated, uncorrelated
