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

"""Classes and functions for ellipse models."""

import logging
import math as maths
import warnings
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, cast, get_args

import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from scipy.optimize import OptimizeResult, minimize
from scipy.special import gamma
from scipy.special import kv as modified_bessel_2nd

from glomar_gridding.constants import DEFAULT_BACKEND, DEFAULT_N_JOBS
from glomar_gridding.distances import mahal_dist_func
from glomar_gridding.types import FForm, ModelType, SuperCategory
from glomar_gridding.utils import deg_to_km

MODEL_TYPE_TO_SUPERCATEGORY: dict[ModelType, SuperCategory] = {
    "ps2006_kks2011_iso": "1_param_matern",
    "ps2006_kks2011_ani": "2_param_matern",
    "ps2006_kks2011_ani_r": "3_param_matern",
    "ps2006_kks2011_iso_pd": "1_param_matern_pd",
    "ps2006_kks2011_ani_pd": "2_param_matern_pd",
    "ps2006_kks2011_ani_r_pd": "3_param_matern_pd",
}

FFORM_TO_MODELTYPE: dict[FForm, ModelType] = {
    "anisotropic_rotated": "ps2006_kks2011_ani_r",
    "anisotropic": "ps2006_kks2011_ani",
    "isotropic": "ps2006_kks2011_iso",
    "anisotropic_rotated_pd": "ps2006_kks2011_ani_r_pd",
    "anisotropic_pd": "ps2006_kks2011_ani_pd",
    "isotropic_pd": "ps2006_kks2011_iso_pd",
}

SUPERCATEGORY_PARAMS: dict[SuperCategory, OrderedDict[str, str]] = {
    "3_param_matern": OrderedDict(
        [
            ("Lx", "degrees"),
            ("Ly", "degrees"),
            ("theta", "radians"),
            ("standard_deviation", "K"),
            ("qc_code", "1"),
            ("number_of_iterations", "1"),
        ]
    ),
    "2_param_matern": OrderedDict(
        [
            ("Lx", "degrees"),
            ("Ly", "degrees"),
            ("standard_deviation", "K"),
            ("qc_code", "1"),
            ("number_of_iterations", "1"),
        ]
    ),
    "1_param_matern": OrderedDict(
        [
            ("R", "degrees"),
            ("standard_deviation", "K"),
            ("qc_code", "1"),
            ("number_of_iterations", "1"),
        ]
    ),
    "3_param_matern_pd": OrderedDict(
        [
            ("Lx", "km"),
            ("Ly", "km"),
            ("theta", "radians"),
            ("standard_deviation", "K"),
            ("qc_code", "1"),
            ("number_of_iterations", "1"),
        ]
    ),
    "2_param_matern_pd": OrderedDict(
        [
            ("Lx", "km"),
            ("Ly", "km"),
            ("standard_deviation", "K"),
            ("qc_code", "1"),
            ("number_of_iterations", "1"),
        ]
    ),
    "1_param_matern_pd": OrderedDict(
        [
            ("R", "km"),
            ("standard_deviation", "K"),
            ("qc_code", "1"),
            ("number_of_iterations", "1"),
        ]
    ),
}

FFORM_PARAMETERS: dict[str, dict[str, Any]] = {
    "isotropic": {
        "n_params": 1,
        "default_guesses": [7.0],
        "default_bounds": [(0.5, 50.0)],
    },
    "isotropic_pd": {
        "n_params": 1,
        "default_guesses": [deg_to_km(7.0)],
        "default_bounds": [(deg_to_km(0.5), deg_to_km(50.0))],
    },
    "anisotropic": {
        "n_params": 2,
        "default_guesses": [7.0, 7.0],
        "default_bounds": [(0.5, 50.0), (0.5, 30.0)],
    },
    "anisotropic_pd": {
        "n_params": 2,
        "default_guesses": [deg_to_km(7.0), deg_to_km(7.0)],
        "default_bounds": [
            (deg_to_km(0.5), deg_to_km(50.0)),
            (deg_to_km(0.5), deg_to_km(30.0)),
        ],
    },
    "anisotropic_rotated": {
        "n_params": 3,
        "default_guesses": [7.0, 7.0, 0.0],
        "default_bounds": [
            (0.5, 50.0),
            (0.5, 30.0),
            (-0.5 * np.pi, 0.5 * np.pi),
        ],
    },
    "anisotropic_rotated_pd": {
        "n_params": 3,
        "default_guesses": [deg_to_km(7.0), deg_to_km(7.0), 0.0],
        "default_bounds": [
            (deg_to_km(0.5), deg_to_km(50.0)),
            (deg_to_km(0.5), deg_to_km(30.0)),
            (-0.5 * maths.pi, 0.5 * maths.pi),
        ],
    },
}


class EllipseModel:
    """
    The class that contains variogram/ellipse fitting methods and parameters

    This class assumes your input to be a standardised correlation matrix
    They are easier to handle because stdevs in the covariance function become 1

    Parameters
    ----------
    anisotropic : bool
        Should the output be an ellipse? Set to False for circle.
    rotated : bool
        Can the ellipse be rotated. If anisotropic is False this value cannot
        be True.
    physical_distance : bool
        Use physical distances rather than lat/lon distance.
    v : float
        Matern Shape Parameter. Must be > 0.0.
    unit_sigma=True: bool
        When MLE fitting the Matern parameters,
        assuming the Matern parameters themselves
        are normally distributed,
        there is standard deviation within the log likelihood function.

        See Wikipedia entry for Maximum Likelihood under:
        - Continuous distribution, continuous parameter space

        Its actual value is not important
        to the best (MLE) estimate of the Matern parameters.
        If one assumes the parameters are normally distributed,
        the mean (best estimate) is independent of its variance.
        In fact in Karspeck et al 2012 [Karspeck]_, it is simply set to 1
        Eq B1).
        This value can however be computed. It serves a similar purpose as
        the original standard deviation:
        in this case, how the actual observed semivariance disperses
        around the fitted variogram.

        The choice to default to 1 follows Karspeck et al. 2012 [Karspeck]_
    """

    def __init__(
        self,
        anisotropic: bool,
        rotated: bool,
        physical_distance: bool,
        v: float,
        unit_sigma: bool = False,
    ) -> None:
        if v <= 0:
            raise ValueError("'v' must be > 0")
        self.anisotropic = anisotropic
        self.rotated = rotated
        self.physical_distance = physical_distance
        self.v = v
        self.unit_sigma = unit_sigma

        self._get_model_names()
        self.supercategory_params = SUPERCATEGORY_PARAMS[self.supercategory]
        self.supercategory_n_params = len(self.supercategory_params)

        self._get_defaults()

        return None

    def _get_model_names(self) -> None:
        """Determine the fform, model type, and supercategory."""
        if self.rotated and not self.anisotropic:
            raise ValueError("Cannot have an isotropic rotated fform")

        fform_builder: list[str] = (
            ["anisotropic"] if self.anisotropic else ["isotropic"]
        )
        if self.rotated:
            fform_builder.append("rotated")
        if self.physical_distance:
            fform_builder.append("pd")

        fform_str: str = "_".join(fform_builder)
        if fform_str not in get_args(FForm):
            raise ValueError("Could not compute fform value from inputs")

        self.fform: FForm = cast(FForm, fform_str)
        self.model_type: ModelType = FFORM_TO_MODELTYPE[self.fform]
        self.supercategory: SuperCategory = MODEL_TYPE_TO_SUPERCATEGORY[
            self.model_type
        ]

        return None

    def _get_defaults(self) -> None:
        """Get default values for the MaternEllipseModel."""
        if self.anisotropic:

            def cov_ij(X, **params):
                return cov_ij_anisotropic(self.v, 1, X[:, 0], X[:, 1], **params)
        else:

            def cov_ij(X, **params):
                return cov_ij_isotropic(self.v, 1, X, **params)

        params = FFORM_PARAMETERS[self.fform]
        self.n_params: int = params["n_params"]
        self.default_guesses: list[float] = params["default_guesses"]
        self.default_bounds: list[tuple[float, float]] = params[
            "default_bounds"
        ]

        self.cov_ij = cov_ij

    def negative_log_likelihood(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: list[float],
        arctanh_transform: bool = True,
    ) -> float:
        """
        Compute the negative log-likelihood given observed X independent
        observations (displacements) and y dependent variable (the observed
        correlation), and Matern parameters params. Namely does the Matern
        covariance function using params, how close it explains the observed
        displacements and correlations.

        log(LL) = SUM (f (y,x|params) )
        params = Maximise (log(LL))
        params = Minimise (-log(LL)) which is how usually the computer solves it
        assuming errors of params are normally distributed

        There is a hidden scale/standard deviation in
        stats.norm.logpdf(scale, which defaults to 1)
        but since we have scaled our values to covariance to correlation (and
        even used Fisher transform) as part of the function, it can be dropped

        Otherwise, you need to have stdev as the last value of params, and
        should be set to the scale parameter

        Parameters
        ----------
        X : np.ndarray
            Observed displacements to the centre of the ellipse.
        y : np.ndarray
            Observed correlation against the centre of the ellipse.
        params : list[float]
            Ellipse parameters (in the current optimize iteration) or if you
            want to compute the actual negative log-likelihood.
        arctanh_transform : bool
            Should the Fisher (arctanh) transform be used
            This is usually option, but it does make the computation
            more stable if they are close to 1 (or -1; doesn't apply here)

        Returns
        -------
        nLL : float
            The negative log likelihood
        """
        sigma = 1 if self.unit_sigma else params[self.n_params]

        match self.n_params:
            case 1:  # Circle
                kwargs = {"R": params[0]}  # Radius
            case 2:  # Un-rotated Ellipse
                kwargs = {"Lx": params[0], "Ly": params[1]}
            case 3:  # Rotated Ellipse
                kwargs = {"Lx": params[0], "Ly": params[1], "theta": params[2]}
            case _:
                raise ValueError("Unexpected length of self.n_params.")

        y_LL = self.cov_ij(X, **kwargs)

        if arctanh_transform:
            # Warning against arctanh(abs(y) > 1);
            arctanh_threshold = 0.999999
            # arctanh_threshold = 1.0
            max_abs_y = np.max(np.abs(y))
            max_abs_yLL = np.max(np.abs(y_LL))
            if max_abs_y >= arctanh_threshold:
                warn_msg = f"abs(y) >= {arctanh_threshold} detected; "
                warn_msg += f"adjusted to threshold; max(abs(y)) = {max_abs_y}"
                warnings.warn(warn_msg, RuntimeWarning)
                y[np.abs(y) > arctanh_threshold] = (
                    np.sign(y[np.abs(y) > arctanh_threshold])
                    * arctanh_threshold
                )
                # y[np.abs(y) > 1] = np.sign(y[np.abs(y) > 1]) * 0.9999

            # if np.any(np.isclose(np.abs(y), 1.0)):
            #     warn_msg = (
            #         "abs(y) is close to 1; max(abs(y))="
            #         + str(max_abs_y)
            #     )
            #     warnings.warn(warn_msg, RuntimeWarning)
            #     y[np.isclose(np.abs(y), 1.0)] = (
            #         np.sign(y[np.isclose(np.abs(y), 1.0)]) * 0.9999
            #     )

            if max_abs_yLL >= 1:
                warn_msg = f"abs(y_LL) >= {arctanh_threshold} detected; "

                warn_msg += (
                    f"adjusted to threshold; max(abs(y_LL))={max_abs_yLL}"
                )
                warnings.warn(warn_msg, RuntimeWarning)
                y_LL[np.abs(y_LL) > arctanh_threshold] = (
                    np.sign(y_LL[np.abs(y_LL) > arctanh_threshold])
                    * arctanh_threshold
                )
                # y_LL[np.abs(y_LL) > 1] = (
                #     np.sign(y_LL[np.abs(y_LL) > 1]) * 0.9999
                # )

            # if np.any(np.isclose(np.abs(y_LL), 1.0)):
            #     warn_msg = (
            #         "abs(y_LL) close to 1 detected; max(abs(y_LL))="
            #         + str(max_abs_yLL)
            #     )
            #     warnings.warn(warn_msg, RuntimeWarning)
            #     y_LL[np.isclose(np.abs(y_LL), 1.0)] = (
            #         np.sign(y_LL[np.isclose(np.abs(y_LL), 1.0)]) * 0.9999
            #     )

            nLL = -1.0 * np.sum(
                stats.norm.logpdf(
                    np.arctanh(y),
                    loc=np.arctanh(y_LL),
                    scale=sigma,
                )
            )
        else:
            nLL = -1.0 * np.sum(stats.norm.logpdf(y, loc=y_LL, scale=sigma))
        return nLL

    def negative_log_likelihood_function(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Callable[[list[float]], float]:
        """
        Creates a function that can be fed into `scipy.optimize.minimize`

        Parameters
        ----------
        X : numpy.ndarray
            (Predictors) Displacements for the training data
        y : numpy.ndarray
            (Response) Correlation values for the training data

        Returns
        -------
        f : Callable[[list[float]], float]
            Negative log-likelihood function for the training data that takes a
            set of parameters and returns a negative log-likelihood value. For
            use in `scipy.optimize.minimize`.
        """

        def f(params: list[float]) -> float:
            return self.negative_log_likelihood(
                X,
                y,
                params,
            )

        return f

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        guesses: list[float] | None = None,
        bounds: list[tuple[float, float]] | None = None,
        opt_method: str = "Nelder-Mead",
        tol: float | None = None,
        estimate_SE: str | None = "bootstrap_parallel",
        n_sim: int = 500,
        n_jobs: int = DEFAULT_N_JOBS,
        backend: str = DEFAULT_BACKEND,
        random_seed: int = 1234,
    ) -> tuple[OptimizeResult, float | None, list[tuple[float, float]]]:
        """
        Default solver in Nelder-Mead as used in Karspeck et al. 2012
        [Karspeck]_
        i.e. https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html
        default max-iter is 200 x (number_of_variables)
        for 3 variables (Lx, Ly, theta) --> 200x3 = 600
        note: unlike variogram fitting, no nugget, no sill, and no residue
        variance (normalised data but Fisher transform needed?)
        can be adjusted using "maxiter" within "options" kwargs

        Much of the variable names are defined the same way as earlier

        Parameters
        ----------
        X : numpy.ndarray
            Array of displacements. Expected to be 1-dimensional if the ellipse
            model is not anisotropic, 2-dimensional otherwise. In units of km if
            the ellipse uses physical distances, otherwise in degrees. The
            displacements are from each position within the test region to the
            centre of the ellipse.
        y : numpy.ndarray
            Vector of observed correlations between the centre of the ellipse
            and each test point.
        guesses=None : list[float] | None
            List of initial values to scipy.optimize.minimize, default guesses
            for the ellipse model are used if not set.
        bounds=None : list[tuple[float, float]] | None
            Tuples/lists of bounds for fitted parameters. Default bounds for
            the ellipse model are used if not set.
        opt_method : str
            scipy.optimize.minimize optimisation method. Defaults to
            "Nelder-Mead". See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
            for valid values.
        tol=None : float | None
            scipy.optimize.minimize convergence tolerance
        estimate_SE='bootstrap_parallel' : str | None
            How to estimate standard error if needed. If not set no standard
            error is computed.
        n_sim=500 : int
            Number of bootstrap to estimate standard error
        n_jobs=DEFAULT_N_JOBS : int
            Number of threads for bootstrapping if `estimate_SE` is set to
            "bootstrap_parallel".
        backend=DEFAULT_BACKEND : str
            joblib backend for bootstrapping.
        random_seed=1234 : int
            Random seed for bootstrap

        Returns
        -------
        results : OptimizeResult
            Output of scipy.optimize.minimize
        SE : float | None
            Standard error of the fitted parameters
        bounds : list[tuple[float, ...]]
            Bounds of fitted parameters
        """
        guesses = guesses or self.default_guesses
        bounds = bounds or self.default_bounds

        if (not self.unit_sigma) and len(guesses) != self.n_params + 1:
            guesses.append(0.1)
            bounds.append((0.0001, 0.5))

        LL_observedXy_unknownparams = self.negative_log_likelihood_function(
            X, y
        )

        # logging.debug(f"X range: {np.min(X)}, {np.max(X)}")
        # logging.debug(f"y range: {np.min(y)}, {np.max(y)}")
        # zipper = zip(guesses, bounds)
        # for g, b in zipper:
        #     logging.debug(f"init guess: {g}, bounds: {b}")

        results: OptimizeResult = minimize(
            LL_observedXy_unknownparams,
            guesses,
            bounds=bounds,
            method=opt_method,
            tol=tol,
        )

        # This does not account for standard errors in the
        # correlation/covariance matrix!
        if estimate_SE is None:
            logging.info("Standard error estimates not required")
            return results, None, bounds

        match estimate_SE:
            case "bootstrap_serial":
                # Serial
                sim_params = np.array(
                    [
                        self._bootstrap_once(
                            X,
                            y,
                            guesses,
                            bounds,
                            opt_method,
                            tol=tol,
                            seed=random_seed + worker,
                        )
                        for worker in range(n_sim)
                    ]
                )
            case "bootstrap_parallel":
                # Parallel
                # On JASMIN Jupyter: n_jobs = 5 leads to 1/3 wallclock time
                kwargs_0 = {"n_jobs": n_jobs, "backend": backend}
                workers = range(n_sim)
                sim_params = Parallel(**kwargs_0)(
                    delayed(self._bootstrap_once)(
                        X,
                        y,
                        guesses,
                        bounds,
                        opt_method,
                        tol=tol,
                        seed=random_seed + worker,
                    )
                    for worker in workers
                )
                sim_params = np.array(sim_params)
            case "hessian":
                # note: autograd does not work with scipy's Bessel functions
                raise NotImplementedError(
                    "Second order deriviative (Hessian) of "
                    + "Fisher Information not implemented"
                )
            case _:
                raise ValueError(f"Unknown estimate_SE value: {estimate_SE}")

        SE = np.std(sim_params, axis=0)

        return results, SE, bounds

    def _bootstrap_once(
        self,
        X: np.ndarray,
        y: np.ndarray,
        guesses: list[float],
        bounds: list[tuple[float, ...]],
        opt_method: str,
        tol: float | None = None,
        seed: int = 1234,
    ) -> list[float]:
        """
        Bootstrap refit the Matern parameters

        Parameters
        ----------
        X : numpy.ndarray
            (Predictors) Displacements for the training data.
        y : numpy.ndarray
            (Response) Correlation values for the training data.
        guesses : list[float],
            Initial values for the parameters.
        bounds : list[tuple[float, ...]],
            Bounds on the parameters.
        opt_method : str,
            Optimisation method.
        tol : float | None = None,
            Tolerance value for the optimiser.
        seed : int = 1234,
            Random seed to use for choosing the subset of the training data to
            use for the bootstrap.

        Returns
        -------
        params : list[float]
            The parameters resulting from a single bootstrap of the optimiser.
        """
        rng = np.random.RandomState(seed)
        len_obs = len(y)
        i_obs = np.arange(len_obs)
        bootstrap_i = rng.choice(i_obs, size=len_obs, replace=True)
        X_boot = X[bootstrap_i, ...]
        y_boot = y[bootstrap_i]
        LL_boot_simulated_params = self.negative_log_likelihood_function(
            X_boot, y_boot
        )
        result: OptimizeResult = minimize(
            LL_boot_simulated_params,
            guesses,
            bounds=bounds,
            method=opt_method,
            tol=tol,
        )
        return result.x


def cov_ij_anisotropic(
    v: float,
    stdev: float,
    delta_x: np.ndarray,
    delta_y: np.ndarray,
    Lx: float,
    Ly: float,
    stdev_j: float | None = None,
    theta: float | None = None,
) -> np.ndarray:
    """
    Covariance structure between base point i and j
    Assuming local stationarity or slowly varying
    so that some terms in PS06 drops off (like Sigma_i ~ Sigma_j instead of
    treating them as different) (aka second_term below)
    this makes formulation a lot more simple
    We let stdev_j opens to changes,
    but in pracitice, we normalise everything to correlation so
    stdev == stdev_j == 1

    Parameters
    ----------
    v : float
        Matern shape parameter
    stdev : float
        Standard deviation at the centre of the ellipse
    delta_x, delta_y : float
        Displacements to remote point as in: (delta_x) i + (delta_y) j in old
        school vector notation
    Lx, Ly : float
        Lx, Ly scale (km or degrees)
    stdev_j : float | None
        Standard deviation, remote point. If set to None, then 'stdev' is used.
    theta : float | None
        Rotation angle of the ellipse in radians.

    Returns
    -------
    cov_ij : float
        Covariance/correlation between local and remote point given displacement
        and Matern covariance parameters
    """
    stdev_j = stdev_j or stdev

    # sigma = sigma_rot_func(Lx, Ly, theta)
    tau = mahal_dist_func(delta_x, delta_y, Lx, Ly, theta=theta)

    first_term = (stdev * stdev_j) / (gamma(v) * (2.0 ** (v - 1)))
    # If data is assumed near stationary locally, sigma_i ~ sigma_j same
    # making (sigma_i)**1/4 (sigma_j)**1/4 / (mean_sigma**1/2) = 1.0
    # Treating it the otherwise is a major escalation to the computation
    # See discussion 2nd paragraph in 3.1.1 in Paciroke and Schervish 2006
    # second_term = 1.0
    inner = 2.0 * tau * np.sqrt(v)
    third_term = np.power(inner, v)
    forth_term = modified_bessel_2nd(v, inner)
    return first_term * third_term * forth_term

    # logging.debug(f"{first_term = }, {first_term.shape = }")
    # logging.debug(f"{third_term = }, {third_term.shape = }")
    # logging.debug(f"{forth_term = }, {forth_term.shape = }")
    # logging.debug(f"{cov_ij = }, {cov_ij.shape = }")
    # return cov_ij


def cov_ij_isotropic(
    v: float,
    stdev: float,
    delta: np.ndarray,
    R: float,
    stdev_j: float | None = None,
) -> np.ndarray:
    """
    Isotropic version of cov_ij_anisotropic. This makes the assumption that
    Lx = Ly = R, i.e. that the model is a circle.

    Parameters
    ----------
    v : float
        Matern shape parameter
    stdev : float
        Standard deviation, local point
    delta : float
        Displacements to remote point
    R : float
        Range parameter (km or degrees)
    stdev_j : float
        Standard deviation, remote point

    Returns
    -------
    cov_ij : float
        Covariance/correlation between local and remote point given displacement
        and Matern covariance parameters
    """
    stdev_j = stdev_j or stdev

    tau = np.abs(delta) / R

    inner = 2.0 * tau * np.sqrt(v)
    first_term = (stdev * stdev_j) / (gamma(v) * (2.0 ** (v - 1)))
    third_term = (inner) ** v
    forth_term = modified_bessel_2nd(v, inner)
    cov_ij = first_term * third_term * forth_term
    return cov_ij
