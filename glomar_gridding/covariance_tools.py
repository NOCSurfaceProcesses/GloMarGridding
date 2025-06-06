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

    3. EOF chop-off -
    :py:func:`glomar_gridding.covariance_tools.eof_chop`:

        Set a target explained variance (say 95%) for the empirical orthogonal
        functions, compute the eigenvalues and eigenvectors up to that explained
        variance. Reconstruct the covariance keeping only EOFs up to the target.
        This is very close to 2, but it reduces the total variance of the
        covariance matrix. The original method requires solving for ALL
        eigenvectors which may not be possible for massive matrices
        (40000x40000 square matrices). This is currently done for the MAT
        covariance matrices which have very large dominant modes.

        Note, this may not adjust negative eigenvalues to be positive.

    4. Other methods not implemented here

        a. shrinkage methods
            https://scikit-learn.org/stable/modules/covariance.html
        b. reprojection (aka Higham's method) [Higham]_
            https://github.com/mikecroucher/nearest_correlation

            https://nhigham.com/2013/02/13/the-nearest-correlation-matrix/

Author S Chan.

Modified by J. Siddons.
"""

from itertools import accumulate
from typing import Any, Literal
import numpy as np
import scipy as sp
from statsmodels.stats import correlation_tools
from warnings import warn


def check_symmetric(
    a: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """Helper function for perturb_sym_matrix_2_positive_definite"""
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


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
    glomar_gridding.covariance_tool.simple_clipping

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
    statsmodels.stats.correlation_tools.cov_nearest

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
        threshold = 5.0 * finfo.resolution * np.max(np.abs(all_eigval))
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
            "threshold must either be number, auto or statsmodels_default"
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
    matrix: np.ndarray,
    atol: float = 1e-5,
) -> np.ndarray:
    """Set small values (abs(x) < atol) in an matrix to 0"""
    cleaned = matrix.copy()
    cleaned[np.abs(matrix) < atol] = 0.0
    return cleaned


def _find_index_explained_variance(eigvals, target=0.95):
    """
    Find the index of the eigenvalue for which the normalised cumulative sum
    exceeds a target variance.
    """
    total_variance = np.sum(eigvals)
    target_explained_variance = target * total_variance
    print(f"{total_variance = }")
    print(f"{target_explained_variance = }")
    csum, i2goal = csum_up_to_val(eigvals, target_explained_variance)
    if csum <= target_explained_variance:
        raise ValueError("Target Explained Variance not exceeded")
    return i2goal


def _find_index_aspect_ratio(
    eigvals: np.ndarray,
    num_grid_pts: int = 180 * 360,
    num_times: int = 41 * 6,
) -> int:
    """
    Defaults are based on:
        41 years ESA data
        6 pentads per month
        and 37000ish 1x1 deg grid points
    Resulting in q ~ 150, threshold ~ 175

    For 5x5 data and 40-ish year of observations
    Observations are monthly:
        q ~ 65, threshold ~ 82

    These parameters do not work in general must be determined from input data.
    """
    _, threshold = _estimate_threshold(num_grid_pts, num_times)
    return -int(np.sum(eigvals > threshold))


def _estimate_threshold(
    num_grid_pts: int = 180 * 360,
    num_times: int = 41 * 6,
) -> tuple[float, float]:
    """
    See 7.2.2 in https://doi.org/10.1016/j.physrep.2016.10.005
    Eigenvalue threshold: threshold = (1.0 + SQRT(q))**2
    Below calculates q and threshold
    """
    q = num_grid_pts / num_times
    if q < 1.0:
        q = 1.0 / q
    threshold = (1.0 + np.sqrt(q)) ** 2.0
    return q, threshold


def eigenvalue_clip(
    cov: np.ndarray,
    method: Literal["explained_variance", "Laloux_2000"] = "explained_variance",
    method_parms={"target": 0.95},
) -> np.ndarray:
    """
    Denoise symmetric damaged covariance/correlation matrix cov by clipping
    eigenvalues

    Explained variance or aspect ratio based threshold
    Aspect ratios is based on dimensionless parameters
    (number of independent variable and observation size)

    .. math::
        q = N/T = (num of independent variable)
        / (num of observation per independent variable)

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

    Returns
    -------
    cov_adj : numpy.ndarray
        Adjusted covariance matrix.
    """
    eigvals, eigvecs = np.linalg.eigh(cov)
    print(f"top 5 eigenvalues = {eigvals[:5]}")
    print(f"bottom 5 eigenvalues = {eigvals[-5:]}")
    n_eigvals = len(eigvals)

    match method:
        case "explained_variance":
            keep_i = _find_index_explained_variance(eigvals, **method_parms)
        case "Laloux_2000":
            keep_i = _find_index_aspect_ratio(eigvals, **method_parms)
        case _:
            raise ValueError("Unknown clipping method")

    clip_i = n_eigvals + keep_i  # Note i2keep is NEGATIVE
    print(f"Number of kept eigenvalues = {-keep_i}")
    print(f"Number of clipped eigenvalues = {clip_i}")

    # The total variance should be preserved after clipping
    # within precision error of the eigenvalues which is
    # O(Max(Eig) * float_accuracy)
    total_var = np.sum(eigvals)
    var_explained_by_i2keep = np.sum(eigvals[keep_i:])
    unexplained_var = total_var - var_explained_by_i2keep
    avg_eigenvals_4_unexplained = unexplained_var / clip_i

    # Find eigenvectors associated up to i2keep
    new_eigvals = eigvals.copy()
    new_eigvals[:keep_i] = avg_eigenvals_4_unexplained
    return eigvecs @ np.diag(new_eigvals) @ eigvecs.T
