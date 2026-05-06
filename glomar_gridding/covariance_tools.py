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

r"""
When estimating covariance matrices, using ellipse-based methods for example,
the results may not be positive-definite. This can be problematic, for instance
the Kriging equations may not be solvable as the inverse matrix cannot be
computed, or simulated states cannot be constructed using
:py:func:`glomar_gridding.stochastic.scipy_mv_normal_draw`. To account for this,
`glomar_gridding` includes a few tools for *fixing* these covariance matrices.
In general, these methods attempt to coerce the input matrix to be
positive-definite by updating the eigenvalues and re-computing the matrix from
these updated eigenvalues and the original eigenvectors. The indicators of an
invalid covariance matrix include

    1. Un-invertible covariance matrices with 0 eigenvalues
    2. Covariance matrices with eigenvalues less than zero

This can typically be a consequence of

    1. Multicollinearity:
       but nearly all very large cov matrices will have rounding errors to have
       this occur
    2. Number of spatial points >> length of time series
       (for ESA monthly pentads: this ratio is about 150)
    3. Covariance is estimated using partial data

In most cases, the most likely causes are 2 and 3.

There are a number of methods included in this module. In general, the approach
is to adjust the eigenvalues to ensure small or negative eigenvalues are
increased to some minimum threshold. The covariance matrix is then re-calculated
using these modified eigenvalues and the original eigenvectors.

In general, the recommended approach is Original Clipping, see
:py:func:`glomar_gridding.covariance_tools.eigenvalue_clip`.

Fixes:

    1. Simple clipping -
    :py:func:`glomar_gridding.covariance_tools.simple_clipping`:

        Cut off the negative, zero, and small positive eigenvalues; this is
        method used in statsmodels.stats.correlation_tools but the version here
        has better thresholds based on the accuracy of the eigenvalues, plus a
        iterative version which is slower but more stable with big matrices. The
        iterative version is recommended for SST/MAT covariances.

        This is used for SST covariance matrices which have less dominant modes
        than MAT; it also preserves more noise.

        Trace (aka total variance) of the covariance matrix is not conserved,
        but it is less disruptive than EOF chop off (method 3).

        It is more difficult to use for covariance matrices with one large
        dominant mode because that raises the bar of accuracy of the
        eigenvalues, which requires clipping off a lot more eigenvectors.

        Note, this will adjust all negative eigenvalues to be positive.

    2. Original clipping -
    :py:func:`glomar_gridding.covariance_tools.eigenvalue_clip`:

        Determine a noise eigenvalue threshold and replace all eigenvalues below
        using the average of them, preserving the original trace (aka total
        variance) of the covariance matrix, but this will require a full
        computation of all eigenvectors, which may be slow and cause memory
        problems

        Note, this will adjust all negative eigenvalues to be positive.

Other methods not implemented here -

    a. shrinkage methods
        https://scikit-learn.org/stable/modules/covariance.html
    b. reprojection (aka Higham's method) [Higham]_
        https://github.com/mikecroucher/nearest_correlation

        https://nhigham.com/2013/02/13/the-nearest-correlation-matrix/

Author S. Chan.

Modified by J. Siddons.
"""

from itertools import accumulate
from typing import Any, Literal
from warnings import warn

import numpy as np
import scipy as sp
from statsmodels.stats import correlation_tools

from glomar_gridding.utils import cor_2_cov, cov_2_cor


def check_symmetric(
    mat: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """
    Check that a matrix is symmetric, within some relative or absolute tolerance
    values.

    Parameters
    ----------
    mat : numpy.ndarray
        The matrix to check.
    rtol : float
        The relative tolerance value.
    atol : float
        The absolute tolerance value.

    Returns
    -------
    bool
        Indicating if the input matrix is symmetric within the tolerances.
    """
    return np.allclose(mat, mat.T, rtol=rtol, atol=atol)


def perturb_cov_to_positive_definite(
    cov: np.ndarray,
    threshold: float | Literal["auto"] = 1e-15,
) -> np.ndarray:
    """
    Force an estimated covariance matrix to be positive definite using the
    eigenvalue clipping with statsmodels.stats.correlation_tools.cov_nearest
    function.

    Deprecated in favour of
    :py:func:`glomar_gridding.covariance_tools.simple_clipping`.

    Parameters
    ----------
    cov : numpy.ndarray
        The estimated covariance matrix that is not positive definite.
    threshold : float | 'auto'
        Eigenvalues below this value are set to 0. If the input is 'auto' then
        the value is determined using the floating-point precision and magnitude
        of the largest eigenvalues.

    Returns
    -------
    cov_adj : numpy.ndarray
        Adjusted covariance matrix

    See Also
    --------
    :py:func:`glomar_gridding.covariance_tool.simple_clipping`

    Notes
    -----
    Other methods:

        - https://nhigham.com/2021/02/16/diagonally-perturbing-a-symmetric-matrix-to-make-it-positive-definite/
        - https://nhigham.com/2013/02/13/the-nearest-correlation-matrix/
        - https://academic.oup.com/imajna/article/22/3/329/708688
    """
    warn(
        "This function is deprecated in favour of "
        + "'glomar_gridding.covariance_tool.simple_clipping'",
        DeprecationWarning,
    )
    matrix_dim = cov.shape
    if (
        (len(matrix_dim) != 2)
        or (matrix_dim[0] != matrix_dim[1])
        or not check_symmetric(cov)
    ):
        raise ValueError("Matrix is not square and/or symmetric.")

    eigenvalues = np.linalg.eigvalsh(cov)
    if threshold == "auto":
        finfo = np.finfo(eigenvalues.dtype)
        threshold = 5.0 * finfo.resolution * np.max(np.abs(eigenvalues))
    if not isinstance(threshold, (float, int)):
        raise TypeError("`threshold` must be numeric, or 'auto'.")

    min_eigen = np.min(eigenvalues)
    max_eigen = np.max(eigenvalues)
    n_negatives = np.sum(eigenvalues < 0.0)
    print("Number of eigenvalues = ", len(eigenvalues))
    print("Number of negative eigenvalues = ", n_negatives)
    print("Largest eigenvalue  = ", max_eigen)
    print("Smallest eigenvalue = ", min_eigen)
    if min_eigen >= 0.0:
        print("Matrix is already positive (semi-)definite.")
        return cov
    cov_adj = correlation_tools.cov_nearest(
        cov,
        return_all=False,
        threshold=threshold,
    )
    if not isinstance(cov_adj, np.ndarray):
        raise TypeError(
            "Output of correlation_tools.cov_nearest is not a numpy array"
        )

    eigenvalues_adj = np.linalg.eigvalsh(cov_adj)
    min_eigen_adj = np.min(eigenvalues_adj)
    max_eigen_adj = np.max(eigenvalues_adj)
    n_negatives_adj = np.sum(eigenvalues_adj < 0.0)
    print("Post adjustments:")
    print("Number of negative eigenvalues (post_adj) = ", n_negatives_adj)
    print("Largest eigenvalue (post_adj)  = ", max_eigen_adj)
    print("Smallest eigenvalue (post_adj) = ", min_eigen_adj)
    return cov_adj


def simple_clipping(
    cov,
    threshold: float | Literal["auto", "statsmodels_default"] = "auto",
    method: Literal["iterative", "direct"] = "iterative",
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    A modified version of:
    https://www.statsmodels.org/dev/generated/statsmodels.stats.correlation_tools.corr_nearest.html

    Force an estimated covariance matrix to be positive definite using the
    eigenvalue clipping with statsmodels.stats.correlation_tools.cov_nearest
    function.

    This is appropriate for covariance matrices which have less dominant modes;
    it also preserves more noise.

    Trace (aka total variance) of the covariance matrix is not conserved,
    but it is less disruptive than EOF chop off.

    Parameters
    ----------
    cov : numpy.ndarray
        The estimated covariance matrix that is not positive definite.
    threshold : float | 'auto'
        Eigenvalues below this value are set to 0. If the input is 'auto' then
        the value is determined using the floating-point precision and magnitude
        of the largest eigenvalues.

    Returns
    -------
    cov_adj : numpy.ndarray
        Adjusted covariance matrix
    summary_dict : dict[str, Any]
        A dictionary containing a summary of the input and results with the
        following keys:

            - "threshold"
            - "smallest_eigv"
            - "determinant"
            - "total_variance"

    See Also
    --------
    :py:func:`statsmodels.stats.correlation_tools.cov_nearest`

    Notes
    -----
    Other methods:

        - https://nhigham.com/2021/02/16/diagonally-perturbing-a-symmetric-matrix-to-make-it-positive-definite/
        - https://nhigham.com/2013/02/13/the-nearest-correlation-matrix/
        - https://academic.oup.com/imajna/article/22/3/329/708688
    """
    n = cov.shape[0]
    all_eigval = np.linalg.eigvals(cov)
    all_eigval = np.sort(all_eigval)
    max_eigval = np.max(all_eigval)
    min_eigval = np.min(all_eigval)
    sum_eigval = np.sum(all_eigval)
    p90_index = int(0.1 * len(all_eigval))
    sumtop10_eigval = np.sum(all_eigval[-p90_index:])
    top10_explained_var = 100.0 * (sumtop10_eigval / sum_eigval)
    print("Pre-adjusted eigenvalue summary")
    print("Largest=", max_eigval)
    print("Smallest/most negative=", min_eigval)
    print("Sum=", sum_eigval)
    print("Explained variance from top 10%=", top10_explained_var, "%")

    if threshold == "auto":
        # According to
        # https://stackoverflow.com/questions/13891225/precision-of-numpys-eigenvaluesh
        # https://www.netlib.org/lapack/lug/node89.html
        #
        # Accuracy of eigenvalue of lapack is greater or equal to
        # MAX(ABS(eigenvalues)) x floating_pt_accuracy of the float type
        #
        # e.g. for float32 np.array:
        # np.finfo(np.float32) -->
        # finfo(
        #     resolution=1e-06,
        #     min=-3.4028235e38,
        #     max=3.4028235e38,
        #     dtype=float32,
        # )
        # Typical SSTA covariance matrices has max eigv ~ 1000 degC**2
        # This gives a lower bound threshold on the order of 1E-6 x 1E3 ~ 1E-3
        #
        # Give some margin of safety to above approximation: we will do 5x of
        # the above
        # For most climate science applications,
        # these eigenvalues are essentially noise to the data
        finfo = np.finfo(all_eigval.dtype)
        threshold = float(5.0 * finfo.resolution * np.max(np.abs(all_eigval)))
    elif threshold == "statsmodels_default":
        # 1e-15 is the precision for np.float64
        # This is the default used in
        # https://www.statsmodels.org/dev/generated/statsmodels.stats.correlation_tools.corr_clipped.html
        #
        # It is NOT SUITABLE based on guidelines to the precision of
        # lapack eigenvalue decomposition, nor the input data can be assumed
        # to be float64!
        threshold = 1e-15
    if not isinstance(threshold, (float, int)):
        raise TypeError(
            "threshold must either be number, auto or statsmodels_default. "
            + f"Got {threshold = }."
        )

    n_negative = int(np.sum(all_eigval < threshold))
    if n_negative == n:
        warn("Input has all negative eigenvalues")
    print("Minimum eigenvalue threshold = ", threshold)
    print("Estimated number of eigenvalues below threshold = ", n_negative)
    n_vec = n_negative

    print(
        "Computing eigenvalues and eigenvector up to the estimated number: "
        + str(n_vec)
    )
    current_eigv, current_eigV = sp.linalg.eigh(
        cov, eigvals_only=False, subset_by_index=[0, n_vec - 1]
    )
    print(current_eigv)
    print(current_eigV)
    print(current_eigV.shape)

    # Make a copy
    cov_adj = cov.copy()

    # Rank-n_vec update
    print("Fixing matrix by reverse clipping")
    if method == "iterative":
        for iii in range(n_vec):
            if current_eigv[iii] > threshold:
                warning_msg = "New estimate of eigenvalue is below threshold,"
                warning_msg += "possibly due to precision; bypassing."
                print((iii, n_vec - 1, current_eigv[iii], warning_msg))
                continue
            worst_eigV = current_eigV[:, iii][np.newaxis]
            VbadxVbadT = worst_eigV * worst_eigV.T
            # This is only if threshold == 0
            # r_peturb = VbadxVbadT * current_eigv[iii]
            # cov_arr_adj = cov_arr_adj - r_peturb
            r_peturb = VbadxVbadT * (threshold - current_eigv[iii])
            cov_adj = cov_adj + r_peturb
            print(
                (
                    iii,
                    n_vec - 1,
                    current_eigv[iii],
                    threshold - current_eigv[iii],
                    np.max(np.diag(r_peturb)),
                )
            )
    elif method == "direct":
        dL = threshold - all_eigval[:n_vec]
        if np.any(dL < 0.0):
            # This might be a problem when eigenvalues are re-estimated
            print("Some new estimates to eigenvalue are above threshold.")
            print("No adjustments will be made for those eigenvalues")
            dL = np.diag(np.max([np.zeros_like(dL), dL], axis=0))
        else:
            dL = np.diag(dL)
        dC = np.matmul(np.matmul(current_eigV, dL), current_eigV.T)
        cov_adj = cov + dC

    print("Computing adjusted eigenvalues, smallest " + str(n_vec))
    new_eigv = sp.linalg.eigh(
        cov_adj, eigvals_only=True, subset_by_index=[0, n_vec - 1]
    )
    new_min_eigv = np.min(new_eigv)
    print("Eigenvalues that were smaller than threshold:")
    print("[:5] : ", new_eigv[:5])
    print("[-5:]: ", new_eigv[-5:])
    print("Smallest eigenvalue=", new_min_eigv)
    new_det = np.linalg.det(cov_adj)
    print("Determinant=", new_det)
    total_var = np.sum(np.diag(cov_adj))
    meta_dict = {
        "threshold": threshold,
        "smallest_eigv": new_min_eigv,
        "determinant": new_det,
        "total_variance": total_var,
    }
    return (cov_adj, meta_dict)


def csum_up_to_val(
    vals: np.ndarray,
    target: float,
    reverse: bool = True,
    niter: int = 0,
    csum: float = 0.0,
) -> tuple[float, int]:
    """
    Find csum and sample index that target is surpassed. Displays a warning if
    the target is not exceeded or the input `vals` is empty.

    Can provide an initial `niter` and/or `csum` value(s), if working with
    multiple arrays in an iterative process.

    If `reverse` is set, the returned index will be negative and will correspond
    to the index required for the non-reversed array. Reverse is the default.

    Parameters
    ----------
    vals : numpy.ndarray
        Vector of values to sum cumulatively.
    target : float
        Value for which the cumulative sum must exceed.
    reverse : bool
        Reverse the array. The index will be negative.
    niter : int
        Initial number of iterations.
    csum : float
        Initial cumulative sum value.

    Returns
    -------
    csum : float
        The cumulative sum at the index when the target has been exceeded.
    niter : int
        The index of the value that results in the cumulative sum exceeding
        the target.

    Note
    ----
    It is actually faster to compute a full cumulative sum with `np.cumsum` and
    then look for the value that exceeds the target. This is not performed in
    this function.

    Examples
    --------
    >>> vals = np.random.rand(1000)
    >>> target = 301.1
    >>> csum_up_to_val(vals, target)
    """
    if vals.size == 0:
        warn("`vals` is empty")
        return csum, niter
    if len(vals) != vals.size:
        # Not a vector
        raise ValueError("`vals` must be a vector")

    vals = vals[::-1] if reverse else vals

    i = 0
    # accumulate defaults to sum
    for i, csum in enumerate(accumulate(vals, initial=csum), start=0):
        if csum > target:
            i = -i if reverse else i
            return csum, niter + i
    warn("Out of `vals`, target not exceeded.")
    i = -i if reverse else i
    return csum, niter + i


def clean_small(
    mat: np.ndarray,
    atol: float = 1e-5,
) -> np.ndarray:
    """
    Clean a matrix by setting small values below some absolute tolerance
    (abs(val) < atol) to 0.

    Parameters
    ----------
    mat : numpy.ndarray
        The matrix to clean.
    atol : float
        The value below which absolute values in the matrix are set to 0.

    Returns
    -------
    cleaned : numpy.ndarray
        The cleaned matrix with small values set to 0.
    """
    cleaned = mat.copy()
    cleaned[np.abs(mat) < atol] = 0.0
    return cleaned


def _find_index_explained_variance(
    eigvals: np.ndarray,
    target: float = 0.95,
) -> int:
    """
    Find the index of the eigenvalue for which the normalised cumulative sum
    exceeds a target variance. This represents the first eigenvalue to retain
    for clipping.

    Parameters
    ----------
    eigvals : numpy.ndarray
        The eigenvalues.
    target : float
        The target proportion of the total variance. Expected to be between 0
        and 1.

    Returns
    -------
    index : int
        The resulting index. The index of the first eigenvalue to keep.
    """
    if target < 0 or target > 1:
        raise ValueError(
            f"Input 'target' must be between 0, and 1, got {target = }"
        )
    total_variance = np.sum(eigvals)
    target_explained_variance = target * total_variance
    print(f"{total_variance = }")
    print(f"{target_explained_variance = }")
    csum, index = csum_up_to_val(eigvals, target_explained_variance)
    if csum <= target_explained_variance:
        raise ValueError("Target Explained Variance not exceeded")
    return index


def _find_index_aspect_ratio(
    eigvals: np.ndarray,
    num_grid_points: int,
    num_time_points: int,
) -> int:
    r"""
    Find the index of the first eigenvalue to retain for clipping using from
    the _aspect ratio_ of the data.

    This method is used to determine the eigenvalues to retain for Laloux
    clipping: :py:func:`glomar_gridding.covariance_tools.laloux_clip`.

    The range of possible eigenvalues for a observed covariance matrix estimated
    from random (uncorrelated) data is a function of the ratio between the
    number of independent variable (features) and number of observation per
    independent variable/feature.

    This can can be considered to be an _aspect ratio_ (between width and the
    height of the data), which determines the upper (as well as lower) possible
    value that the eigenvalue that will arise just by chance.

    The threshold is defined as:

    .. math::
        \text{threshold} = (1.0 + \sqrt{\text{aspect ratio}})^2

    Note: the aspect ratio is defined so that it always >= 1, so the ratio is
    inverted if it is < 1.

    An example set of parameters:

    - "num_time_points": (41 * 6): 41 years ESA data, 6 pentads per month.
    - "num_grid_points": (360 * 180): Number of grid-points in a global 1-degree
    resolution map.

    Which results in an aspect ratio of approximately 150, threshold
    approximately 175.

    For 5x5 data and 40 years of monthly observations: the aspect ratio is
    approximately 65, and threshold is approximately 82.

    Parameters
    ----------
    eigvals : numpy.ndarray
        The eigenvalues.
    num_grid_points : int
        Number of grid points in the system.
    num_time_points : int
        Number of time-points in the system.

    Returns
    -------
    index : int
        The index of the first eigenvalue to retain, based on the shape of the
        data.

    See Also
    --------
    :py:func:`glomar_gridding.covariance_tools.laloux_clip`.

    References
    ----------
    See 7.2.2 in [Bun]_, Equation 0.3 in [Laloux_2000]_
    """
    aspect_ratio = num_grid_points / num_time_points
    if aspect_ratio < 1.0:
        aspect_ratio = 1.0 / aspect_ratio
    threshold = (1.0 + np.sqrt(aspect_ratio)) ** 2.0
    return -int(np.sum(eigvals > threshold))


def laloux_clip(
    cov: np.ndarray,
    num_grid_pts: int | None = None,
    num_time_pts: int = 40,
) -> np.ndarray:
    r"""
    Estimate the largest eigenvalue that one will get from covariance or
    correlation matrices that are generated by random uncorrelated matrices
    aka the _noise_ level of the eigenvalues.

    Eq 0.3 in [Laloux]_ says eigenvalues of covariances generated by
    uncorrelated random vectors with constant variance have a max of:

    .. math::
        \lambda_{max} = \sigma^2 (1 + Q + 2 \sqrt{Q})

    where:

    .. math::
        Q = \frac{\text{num of features}}{\text{length of each feature}}

    One can make a simple Monte-Carlo simulation to check that (a few lines of
    Python code will do). With variable (but realistic observed) variance,
    equation above is not exactly correct. However, Monte-Carlo simulations of
    :math:`\lambda_{max}` show differences are not large; it becomes a somewhat
    larger, determined by the larger variances of the variable variances, but
    the estimate is correct up to the first significant digit.

    Hence :math:`\lambda_{max}` can be thought of a floor for non-noise EOFs,
    as this is the largest eigenvalue that can be generated by random
    uncorrelated data, and is only a function of Q (aka aspect ratio
    of the number of features with the length of each feature vector).

    Any eigenvalues smaller than :math:`\lambda_{max}` (including negative ones)
    are essentially noise and can possibly be trimmed.

    This approach is aggressive. For climate sciences, Q can be
    large; for example if number of features are on the order of 1000-2000 (a
    global 5x5 grid land-only, sea-only, or of any terrain (up to 2592)) with
    30-40 data points per feature (one value per year), the threshold can be
    reached within first 3-5 EOFs. Q would only worsen if number of features
    increases while number of values per feature decreases (aka higher
    resolution remote sensing data).

    The explained variance approach (default), even if set to a generous
    value (say 70%; see [Wilks]_ chapter 11) would still keep a lot more EOFs.
    Noting that just because some (many) EOFs are noisy or degenerate, they are
    still part of the original covariance; they are actual bits of our best
    estimate to the data. It is desirable to minimise the perturbation applied
    to the covariance, keeping it as close as it is original; noise and
    degeneracy may be interesting by their own right (say in the physical
    interpret the EOFs -- a contentious issue), but that is not our
    primary concern here. See [North]_ Section 5 and [Higham_blog_nearest]_
    for further discussion.

    Above requires q and sigma be the same for all features/independent
    variables. A way to get to around that is standardise the covariance into
    correlation and to apply the trimming to the correlation. The trimmed
    correlation can then converted back to the covariance by putting the
    correct standard deviations back into the diagonal. This effectively
    forces/averages the variance to constant.

    Parameters
    ----------
    cov : numpy.ndarray
        Input covariance matrix to be adjusted to positive definite.

    num_grid_pts : int | None
        Number of spatial grid points for covariance, this should usually be the
        shape of the covariance matrix. If unset, this will default to the size
        of the input covariance matrix (`cov.shape[0]`).

    num_time_pts : int
        Length of the time series that is behind the covariance generation
        Default 40 (aka 40 Jans from 1981-2020), since this is what used
        originally when covariances are generated for modern satellite era
        data (early 1980-ish to 2020-ish). It is important this value to
        be set correctly.

    Returns
    -------
    numpy.ndarray
        Adjusted covariance matrix.

    See Also
    --------
    :py:func:`glomar_gridding.covariance_tools.explained_variance_clip`

    References
    ----------
    - [Laloux]_
    - [Bun]_
    """
    num_grid_pts = num_grid_pts or int(cov.shape[0])
    vars = np.diag(cov)
    cor = cov_2_cor(cov)

    eigvals, eigvecs = np.linalg.eigh(cor)

    keep_i = _find_index_aspect_ratio(
        eigvals,
        num_grid_points=num_grid_pts,
        num_time_points=num_time_pts,
    )

    clipped_cor = _eigenvalue_clip(
        eigvals=eigvals,
        eigvecs=eigvecs,
        keep_i=keep_i,
    )

    return cor_2_cov(clipped_cor, vars)


def explained_variance_clip(
    cov: np.ndarray,
    target_variance_fraction: float = 0.95,
) -> np.ndarray:
    r"""
    Clip all EOFs beyond a certain level of explained variance
    Starting from the EOF with the largest eigenvalue (explained variance)
    descending down the eigenvalues until the cumulative sum of the
    eigenvalues just goes beyond ( Trace(cov) x target_variance_fraction )

    All the larger eigenvalues up to the above thresholds are left unchanged.

    Eigenvalues beyond (smaller positive and negative ones) are clipped instead.
    If n is the first eigenvalue to be clipped, and there are N eigenvalues,
    sorted in descending order:

    .. math::
        (\lambda)_{clipped} = (\lambda_{n}, \lambda_{n+1}, ..., \lambda_{N})

    They now get a revised eigenvalue of E(lambda_clipped).
    Checks are in place within _eigenvalue_clip to ensure the
    threshold is suitable.

    The threshold itself is subjective, but they are based on
    on making intelligent guesses what may be good enough to perturb
    the matrix to make it positive (semi) definite. This approach
    would always give an acceptable and closest stable matrix
    ([Higham_blog_nearest]_) as long as the thresholds
    are set sensibly.

    The automatic method (laloux_clip) (sort-of) does it for you,
    but it only works after the covariance is standardised (or the
    diagonal of the covariance matrix is constant). It is also an
    aggressive method which is based on noisiness and probability
    distribution of eigenvalues. Noting that closeness to the original
    matrix is our primary goal, not the noisiness of eigenvalues.

    - [Jolliffe]_ recommends 70-90% for truncation
    - Graphical guidances ("spectrum of eigenvalues") are helpful ([Wilks]_,
      [Laloux]_)

    Noting that [Wilks]_ and [Jolliffe]_ mostly concern with TRUNCATION not
    clipping. However, clipping is essentially a modified version of
    truncation but with the explained variance fully conserved, nor
    we cannot use covariances based on truncated EOFs; it will have 0
    eigenvalues, many of them will become negative after floating point
    errors.

    95% is somewhat higher than [Jolliffe]_ 70-90% guidance. Operational
    experience with 5x5 monthly data indicates even at 95%, you are
    only retaining 200-400 eigenvalues out of 36 x 72 = 2592 possible
    eigenvalues (if data is global). There are only ~10-20 negative eigenvalues
    plus the 2000-ish eigenvalues that sits out of 95% threshold. The magitude
    of the negative values are 2-plus order of magnitude smaller than the
    largest positive eigenvalues.

    Parameters
    ----------
    cov : numpy.ndarray
        Input covariance matrix to be adjusted to positive definite.

    target_variance_fraction : float
        Starting from the largest eigenvalue and descending, all eigenvalues
        larger than (the trace of cov x target_variance_fraction) are left
        unmodified. Eigenvalues beyond are modified.

    Returns
    -------
    numpy.ndarray
        Adjusted covariance matrix.

    See Also
    --------
    :py:func:`glomar_gridding.covariance_tools.laloux_clip`

    References
    ----------
    - [Laloux]_
    - [Jolliffe]_
    - [Wilks]_
    """
    if not 0.0 < target_variance_fraction <= 1.0:
        raise ValueError("'target_variance_fraction' must be (0, 1.0]")

    eigvals, eigvecs = np.linalg.eigh(cov)

    keep_i = _find_index_explained_variance(
        eigvals,
        target=target_variance_fraction,
    )

    return _eigenvalue_clip(
        eigvals=eigvals,
        eigvecs=eigvecs,
        keep_i=keep_i,
    )


def _eigenvalue_clip(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    keep_i: int,
) -> np.ndarray:
    total_var = np.sum(eigvals)
    var_explained_by_i2keep = np.sum(eigvals[keep_i:])

    print(f"total explained variance = {total_var}")
    print(f"clipped explained variance = {var_explained_by_i2keep}")

    if total_var < var_explained_by_i2keep:
        explained_needed = np.sum(eigvals[keep_i + 1 :])
        new_threshold = explained_needed / total_var
        raise ValueError(
            "Variance explained by retained eigenvalues exceeds total "
            + "variance. Resulting matrix will have negative eigenvalues. "
            + f"Try using a lower threshold. A value below {new_threshold:.2f} "
            + "may work with "
            + "glomar_gridding.covariance_tools.explained_variance_clip."
        )

    print(
        f"New explained variance = {(var_explained_by_i2keep / total_var):.2%}"
    )

    print(f"bottom 5 eigenvalues = {eigvals[:5]}")
    print(f"top 5 eigenvalues = {eigvals[-5:]}")
    n_eigvals = len(eigvals)

    # Ensure keep_i is negative
    keep_i = keep_i if keep_i < 0 else -keep_i

    clip_i = n_eigvals + keep_i  # Note i2keep is NEGATIVE
    print(f"Number of kept eigenvalues = {-keep_i}")
    print(f"Number of clipped eigenvalues = {clip_i}")

    # The total variance should be preserved after clipping
    # within precision error of the eigenvalues which is
    # O(Max(Eig) * float_accuracy)
    unexplained_var = total_var - var_explained_by_i2keep
    avg_eigenvals_4_unexplained = unexplained_var / clip_i

    # Find eigenvectors associated up to i2keep
    new_eigvals = eigvals.copy()
    new_eigvals[:keep_i] = avg_eigenvals_4_unexplained
    out = eigvecs @ np.diag(new_eigvals) @ eigvecs.T

    if not (np.linalg.eigvalsh(out) > 0).all():
        warn(
            "Resulting matrix is not positive-definite, and may not be a "
            + "valid covariance matrix."
        )

    return out


def eigenvalue_clip(
    cov: np.ndarray,
    method: Literal["explained_variance", "Laloux_2000"] = "explained_variance",
    **kwargs,
) -> np.ndarray:
    r"""
    Denoise symmetric damaged covariance/correlation matrix cov by clipping
    eigenvalues

    Explained variance or aspect ratio based threshold
    Aspect ratios is based on dimensionless parameters
    (number of independent variable and observation size)

    .. math::
        q = N/T = \frac{\text{num of independent variable}}{\text{num of
                                          observation per independent variable}}

    Does not give the same results as in eig_clip

    explained_variance here does not have the same meaning.
    The trace of a correlation, by definition, equals the number of diagonal
    elements, which isn't intituatively linked to actual explained variance
    in climate science sense

    This is done by KEEPING the largest explained variance
    in which (number of basis vectors to be kept) >> (number of rows)
    In ESA data, keeping 95% variance means keeping top ~15% of the
    eigenvalues

    Parameters
    ----------
    cov : numpy.ndarray
        Input covariance matrix to be adjusted to positive definite.
    method : "explained_variance" | "Laloux_2000"
        Method used to identify the index of the eigenvalues to clip. If set to
        "explained_variance" then the sorted eigenvalues below the target
        variance are *clipped*. If "Laloux_2000" is set, then the method of
        [Laloux]_ is used.
    **kwargs

    Returns
    -------
    numpy.ndarray
        Adjusted covariance matrix.

    See Also
    --------
    :py:func:`glomar_gridding.covariance_tools.explained_variance_clip`

    See Also
    --------
    :py:func:`glomar_gridding.covariance_tools.laloux_clip`.
    """
    match method:
        case "explained_variance":
            return explained_variance_clip(cov, **kwargs)
        case "Laloux_2000":
            return laloux_clip(cov, **kwargs)
        case _:
            raise ValueError("Unknown clipping method")


def validate_covariance(
    cov: np.ndarray,
    sym_atol: float = 1e-5,
) -> np.ndarray:
    """
    Validate that a covariance is square and symmetric.

    If the input is symmetric return itself. If it is close to symmetric (within
    'sym_atol') then return (cov + cov.T) / 2. Otherwise raise an error.

    Parameters
    ----------
    cov : numpy.ndarray
        Covariance Matrix
    sym_atol : float
        absolute tolerance to check symmetry of cov

    Returns
    -------
    cov : numpy.ndarray
        Symmetric square covariance matrix
    """
    cov_shape = cov.shape
    if len(cov_shape) != 2:
        raise ValueError("cov should be 2D.")
    if cov_shape[0] != cov_shape[1]:
        raise ValueError("cov is not a square matrix")

    if not sp.linalg.issymmetric(cov):
        if sp.linalg.issymmetric(cov, atol=sym_atol):
            warn(
                "cov is nearly symmetric but not exactly so, "
                + "using (cov + cov.T) / 2 instead."
            )
            return (cov + cov.T) / 2
        else:
            raise ValueError("cov is not symmetric.")

    return cov
