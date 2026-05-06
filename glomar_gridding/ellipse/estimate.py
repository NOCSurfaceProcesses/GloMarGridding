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
Class to calculate the covariance (and correlation) of gridded observed data
over time. These values are used to estimate the ellipse parameters with an
instance of `glomar_gridding.ellipse.EllipseModel` as a reference.
"""

import math as maths
from typing import Any
from warnings import warn

import numpy as np
import xarray as xr
from sklearn.metrics.pairwise import haversine_distances

from glomar_gridding.constants import DEFAULT_N_JOBS, RADIUS_OF_EARTH_KM
from glomar_gridding.distances import displacements
from glomar_gridding.ellipse import EllipseModel
from glomar_gridding.types import DeltaXMethod
from glomar_gridding.utils import (
    cov_2_cor,
    is_iter,
    mask_array,
    uncompress_masked,
)


class EllipseBuilder:
    """
    Class to build spatial covariance and correlation matrices used to estimate
    ellipse parameterss using an instance of EllipseModel which sets up the
    defaults for a given configuration.

    To fit ellipse parameters to the correlation of the input data_array, call
    self.fit_ellipse_model.

    Parameters
    ----------
    data_array : numpy.ndarray | numpy.ma.MaskedArray
        Training data stored within a numpy array. In general, this input should
        be extracted from an xarray.DataArray, and masked appropriately.
    coords : xarray.Coordinates
        The coordinates associated with the data_array value. It is expected
        that these are ["time", "latitude", "longitude"]
    """

    def __init__(
        self,
        data_array: np.ndarray,
        coords: xr.Coordinates,
    ) -> None:
        # Defining the input data
        self.data = mask_array(data_array)
        self.coords = coords
        self.xy_shape = self.data[0].shape
        if len(self.xy_shape) != 2:
            raise ValueError(
                "Time slice maps should be 2D; check extra dims (ensemble?)"
            )
        self.big_covar_size = np.prod(self.xy_shape)

        self._parse_coords()
        self._detect_mask()

        # Calculate the actual covariance and correlation matrix:
        self.calc_cov()

        return None

    def _parse_coords(self) -> None:
        # Check input data_cube is actually usable
        self.tcoord_pos = -1
        self.xycoords_pos = []
        self.xycoords_name = []
        for i, coord in enumerate(self.coords):
            if coord == "time":
                self.tcoord_pos = i
            if coord in ["latitude", "longitude"]:
                self.xycoords_pos.append(i)
                self.xycoords_name.append(coord)
        if self.tcoord_pos == -1:
            raise ValueError("Input cube needs a time dimension")
        if self.tcoord_pos != 0:
            raise ValueError("Input cube time dimension not at 0")
        if len(self.xycoords_pos) != 2:
            raise ValueError(
                "Input cube need two spatial dimensions "
                + "('latitude' and 'longitude')"
            )
        self.xycoords_pos = tuple(self.xycoords_pos)

        # Look-up table of the coordinates
        self.xx, self.yy = np.meshgrid(
            self.coords["longitude"].values,
            self.coords["latitude"].values,
        )
        self.xi, self.yi = np.meshgrid(
            np.arange(len(self.coords["longitude"].values)),
            np.arange(len(self.coords["latitude"].values)),
        )
        # Length of time dimension
        self.time_n = len(self.coords["time"].values)

        return None

    def _detect_mask(self) -> None:
        # Detect data mask and determine dimension of array without masked data
        # Almost certain True near the coast
        self.data_has_mask = np.ma.is_masked(self.data)
        if self.data_has_mask:
            # Depending on dataset, the mask might not be invariant
            # (like sea ice)
            # xys with time varying mask are currently discarded.
            # If analysis is conducted seasonally
            # this should normally not a problem unless in high latitudes
            self.mask = np.any(self.data.mask, axis=0)
            self.mask_1D = self.mask.flatten()
            self._self_mask()
            self.small_covar_size = np.sum(np.logical_not(self.mask))
        else:
            self.mask = np.zeros_like(self.data[0], dtype=bool)
            self.mask_1D = self.mask.flatten()
            self.small_covar_size = self.big_covar_size
        self.x_masked = np.ma.masked_where(self.mask, self.xx)
        self.y_masked = np.ma.masked_where(self.mask, self.yy)
        # Add grid indices for convenience
        self.xi_masked = np.ma.masked_where(self.mask, self.xi).compressed()
        self.yi_masked = np.ma.masked_where(self.mask, self.yi).compressed()
        self.xy_masked = np.column_stack(
            [self.x_masked.compressed(), self.y_masked.compressed()]
        )
        self.xy_full = np.column_stack(
            [self.x_masked.flatten(), self.y_masked.flatten()]
        )

        return None

    def calc_cov(
        self,
        rounding: int | None = None,
    ) -> None:
        """
        Calculate covariance and correlation matrices.

        Parameters
        ----------
        rounding : int | None
            Round the values of the output.
        """
        # Reshape data to (t, xy),
        # get rid of mask values -- cannot calculate cov for such data
        xyflatten_data = self.data.reshape((self.time_n, self.big_covar_size))
        xyflatten_data = np.ma.compress_rowcols(xyflatten_data, -1)
        # Remove mean --
        # even data that says "SST anomalies" don't have zero mean (?!)
        xy_mean = np.mean(xyflatten_data, axis=0, keepdims=True)
        xyflatten_data = xyflatten_data - xy_mean
        self.cov = np.matmul(np.transpose(xyflatten_data), xyflatten_data)
        self.cov /= self.time_n - 1
        # xy_cov = _AT_A(xyflatten_data)

        if rounding is not None:
            self.cov = np.round(self.cov, rounding)

        self.cor = cov_2_cor(self.cov, rounding=rounding)

        return None

    def _self_mask(self) -> None:
        """Broadcast cube_mask to all observations"""
        broadcasted_mask = np.broadcast_to(self.mask, self.data.shape)
        self.data = np.ma.masked_where(broadcasted_mask, self.data)

    def fit_ellipse_model(
        self,
        xy_point: int,
        matern_ellipse: EllipseModel,
        max_distance: float = 6000,
        min_distance: float = 0.3,
        delta_x_method: DeltaXMethod | None = "Modified_Met_Office",
        guesses: list[float] | None = None,
        bounds: list[tuple[float, float]] | None = None,
        opt_method: str = "Nelder-Mead",
        tol: float = 0.001,
        estimate_SE: str | None = None,
        n_jobs: int = DEFAULT_N_JOBS,
        n_sim: int = 500,
        physical_distance_selection: bool = True,
    ) -> dict[str, Any] | None:
        """
        Fit ellipses/covariance models using local covariances.

        The form of the covariance model depends on the 'anistropic' and
        'rotated' attributes of the ellipse model:

        - anisotropic=False (radial distance only)
        - anistropic=True and rotated=False (x and y are different, but not
          rotated)
        - anistropic=True and rotated=True (x and y are different, ellipse can
          be rotated)

        The ellipse model attribute v = matern covariance function shape
        parameter.
        Karspeck et al. [Karspeck]_ and Paciorek & Schervish
        [PaciorekSchervish]_
        use 3 and 4 but 0.5 and 1.5 are popular. 0.5 gives an exponential decay:
        lim v-->inf, Gaussian shape

        delta_x_method: only meaningful for physical_distance ellipses:

        - "Met_Office": Cylindrical Earth delta_x = 6400km x delta_lon
          (in radians)
        - "Modified_Met_Office": uses the average zonal dist at different
          lat

        Parameters
        ----------
        xy_point : int
            The index point where ellipses will be fitted to

        max_distance : float
            Maximum separation in distance unit that data will be fed
            into parameter fitting
            Units depend on physical_distance attribute (km if True, otherwise
            degrees).

        min_distance: float
            Minimum separation in distance unit that data
            will be fed into parameter fitting
            Units depend on physical_distance attribute (km if True, otherwise
            degrees).
            Note: Due to the way we compute the Matern function,
            it is undefined at dist == 0 even if the limit -> zero is obvious.

        delta_x_method="Modified_Met_Office": str
            How to compute distances between grid points
            For istropic variogram/covariances, this is a trivial problem;
            you can just take the haversine or
            Euclidean ("tunnel") distance as they are non-directional.

            But it is non trivial for anistropic cases,
            you have to define a set of orthogonal space. In HadSST4,
            Earth is assumed to be cylindrical "tin can" Earth,
            so you can just define the orthogonal space by
            lines of constant lat and lon (delta_x_method="Met_Office").

            The modified "Modified_Met_Office" is a variation to that,
            but allow the tin can get squished at the poles.
            (Sinusoidal projection). This does results in a problem:
            the zonal displacement now depends in which latitude
            you compute on (at the beginning latitude or at the end latitude).
            Here we take the average of the two.

        guesses=None: tuple of floats; None uses default guess values
            Initial guess values that get feeds in the optimizer for MLE.
            In scipy, you are required to do so (but R often doesn't).
            You should anyway; sometimes they do funny things
            if you don't (per recommendation of David Stephenson)

        bounds=None: tuple of floats; None uses default bounds values
            This is essentially a Bayesian "uniformative prior"
            that forces convergence if the optimizer hits the bound.
            For lower resolution fitting, this is rarely a problem.
            For higher resolution fits, this often interacts with
            the limit of the data you can put into the fit the optimizer
            may fail to converge if the input data is very smooth (aka ENSO
            region, where anomalies are smooth over very large (~10000km)
            scales).

        opt_method='Nelder-Mead': str
            scipy.optimize method. Nelder-Mead is the one used by [Karspeck]_.
            See https://docs.scipy.org/doc/scipy/tutorial/optimize.html
            for valid options

        tol=0.001: float
            Set convergence tolerance for scipy optimize.
            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize

            Note on new tol kwarg:
            For N-M, this sets the value to both xatol and fatol
            Default is 1E-4 (?)
            Since it affects accuracy of all values including rotation
            rotation angle 0.001 rad ~ 0.05 deg

        estimate_SE=None : str | None
            The code can estimate the standard error if the Matern parameters.
            This is not usually used or discussed for the purpose of kriging.
            Certain opt_method (gradient descent) can do this automatically
            using Fisher Info for certain covariance function,
            but is not possible for some nasty functions (aka Bessel
            func) gets involved nor it is possible for some optimisers
            (such as Nelder-Mead).
            The code does it using bootstrapping.

        n_jobs=DEFAULT_N_JOBS: int
            If parallel processing, number of threads to use.

        n_sim : int
            Number of simulations to bootstrap for SE estimation.

        physical_distance_selection : bool
            Select training data using physical distance (haversine distance) or
            Euclidean distance of degree difference between each position and
            all possible training data. Data that falls within the min and max
            distance values using the selected distance method are selected as
            training data.

        Returns
        -------
        dict
            Dictionary with results of the fit and the observed correlation
            matrix.
        """
        R2 = uncompress_masked(self.cor[xy_point, :], self.mask_1D).reshape(
            self.xy_shape
        )

        X_train, y_train = self._get_train_data(
            xy_point=xy_point,
            min_distance=min_distance,
            max_distance=max_distance,
            anisotropic=matern_ellipse.anisotropic,
            delta_x_method=delta_x_method,
            physical_distance=matern_ellipse.physical_distance,
            physical_distance_selection=physical_distance_selection,
        )

        if len(y_train) == 0:
            warn(f"No training data for idx {xy_point}")
            return None

        results, SE, bounds = matern_ellipse.fit(
            X_train,
            y_train,
            guesses=guesses,
            bounds=bounds,
            opt_method=opt_method,
            tol=tol,
            estimate_SE=estimate_SE,
            n_jobs=n_jobs,
            n_sim=n_sim,
        )

        model_params = results.x.tolist()

        self._check_params(matern_ellipse, model_params)

        stdev = None
        if not matern_ellipse.unit_sigma:
            stdev = model_params.pop()

        # Meaning of fit_success (int)
        # 0: success
        # 1: success but with one parameter reaching lower boundaries
        # 2: success but with one parameter reaching upper boundaries
        # 3: success with multiple parameters reaching the boundaries
        #    (aka both Lx and Ly), can be both at lower or upper boundaries
        # 9: fail, probably due to running out of maxiter
        #    (see scipy.optimize.minimize kwargs "options)"
        if results.success:
            fit_success = _get_fit_score(model_params, bounds, results.nit)
            # print("RMSE of multivariate norm fit = ", stdev)
        else:
            # print("Convergence fail after ", results.nit, " iterations.")
            # print(model_params)
            fit_success = 9
        # print("QC flag = ", fit_success)

        std_dev = np.sqrt(self.cov[xy_point, xy_point])
        model_params.append(std_dev)
        model_params.append(fit_success)
        model_params.append(results.nit)

        return {
            "Correlation": R2,
            "Results": results,
            "ModelParams": model_params,
            "Success": fit_success,
            "StandardDeviation": std_dev,
            "StandardError": SE,
            "RMSE": stdev,
        }

    def _check_params(
        self,
        ellipse: EllipseModel,
        model_params: list[Any],
    ) -> None:
        """Ensure Lx > Ly, ensure theta is between -pi, pi"""
        # Updates in place
        # Handle Ly > Lx
        if ellipse.anisotropic and model_params[1] > model_params[0]:
            model_params[0], model_params[1] = model_params[1], model_params[0]
            if ellipse.rotated:
                model_params[2] += np.pi / 2

        # Ensure theta is between -pi, pi
        if not ellipse.rotated:
            return None

        if model_params[2] > np.pi:
            model_params[2] -= np.pi
        if model_params[2] <= -np.pi:
            model_params[2] += np.pi
        return None

    def _get_train_data(
        self,
        xy_point: int,
        min_distance: float,
        max_distance: float,
        anisotropic: bool,
        delta_x_method: DeltaXMethod | None,
        physical_distance: bool = True,
        physical_distance_selection: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        if physical_distance and (delta_x_method is None):
            raise ValueError(
                "Cannot have physical_distance with unset delta_x_method"
            )
        lonlat = self.xy_masked[xy_point]
        y = self.cor[xy_point, :]

        disp_y, disp_x = displacements(
            self.xy_masked[:, 1],
            self.xy_masked[:, 0],
            lonlat[1],
            lonlat[0],
            delta_x_method=delta_x_method,
        )

        if delta_x_method is None or not physical_distance_selection:
            if delta_x_method is not None:
                dy, dx = displacements(
                    self.xy_masked[:, 1],
                    self.xy_masked[:, 0],
                    lonlat[1],
                    lonlat[0],
                    delta_x_method=None,
                )
                # dy and dx are in degrees
                deg_distance = np.linalg.norm(np.column_stack([dy, dx]), axis=1)
            else:
                # disp_y and disp_x are in degrees
                deg_distance = np.linalg.norm(
                    np.column_stack([disp_x, disp_y]), axis=1
                )
            valid_dist_idx = np.where(
                (deg_distance <= max_distance)
                & (deg_distance >= min_distance)
                # Delete the origin (can't have dx = dy = 0)
                & (deg_distance != 0)
            )[0]
            y_train = y[valid_dist_idx]

            if anisotropic:
                X_train = np.column_stack([disp_x, disp_y])[valid_dist_idx, :]
                if physical_distance:
                    X_train *= RADIUS_OF_EARTH_KM
                return X_train, y_train

            if physical_distance:
                latlons = np.radians(
                    np.column_stack(
                        [
                            self.xy_masked[valid_dist_idx, 1],
                            self.xy_masked[valid_dist_idx, 0],
                        ]
                    )
                )
                latlon = np.radians(np.array([lonlat[1], lonlat[0]])).reshape(
                    1, -1
                )
                distance = (
                    haversine_distances(latlon, latlons)[0] * RADIUS_OF_EARTH_KM
                )
                return distance, y_train
            return deg_distance[valid_dist_idx], y_train

        # disp_y and disp_x are in radians
        latlons = np.radians(
            np.column_stack([self.xy_masked[:, 1], self.xy_masked[:, 0]])
        )
        latlon = np.radians(np.array([lonlat[1], lonlat[0]])).reshape(1, -1)
        distance = haversine_distances(latlon, latlons)[0] * RADIUS_OF_EARTH_KM
        valid_dist_idx = np.where(
            (distance <= max_distance)
            & (distance >= min_distance)
            # Delete the origin (can't have dx = dy = 0)
            & (distance != 0)
        )[0]
        y_train = y[valid_dist_idx]
        if anisotropic:
            X_train = np.column_stack([disp_x, disp_y])[valid_dist_idx, :]
            return RADIUS_OF_EARTH_KM * X_train, y_train
        return distance[valid_dist_idx], y_train

    def compute_params(
        self,
        default_value: Any,
        matern_ellipse: EllipseModel,
        max_distance: float = 6000,
        min_distance: float = 0.3,
        delta_x_method: DeltaXMethod | None = "Modified_Met_Office",
        guesses: list[float] | None = None,
        bounds: list[tuple[float, float]] | None = None,
        opt_method: str = "Nelder-Mead",
        tol: float = 1e-4,
        estimate_SE: str | None = None,
        n_jobs: int = DEFAULT_N_JOBS,
        n_sim: int = 500,
        physical_distance_selection: bool = True,
    ) -> xr.Dataset:
        """
        Fit ellipses/covariance models using local covariances to all unmasked
        grid points.

        The form of the covariance model depends on the 'anistropic' and
        'rotated' attributes of the ellipse model:

        - anisotropic=False (radial distance only)
        - anistropic=True and rotated=False (x and y are different, but not
          rotated)
        - anistropic=True and rotated=True (x and y are different, ellipse can
          be rotated)

        The ellipse model attribute v = matern covariance function shape
        parameter.
        Karspeck et al. [Karspeck]_ and Paciorek & Schervish
        [PaciorekSchervish]_
        use 3 and 4 but 0.5 and 1.5 are popular. 0.5 gives an exponential decay:
        lim v-->inf, Gaussian shape

        delta_x_method: only meaningful for physical_distance ellipses:

        - "Met_Office": Cylindrical Earth delta_x = 6400km x delta_lon
          (in radians)
        - "Modified_Met_Office": uses the average zonal dist at different
          lat

        Parameters
        ----------
        default_value : Any
            Default value(s) to fill arrays where parameter estimation is not
            possible (typically due to masking). Typically, one should set a
            value that is appropriate to the type of the field. If a single
            value is provided, this is used for all fields. If not, the length
            of the list of default values must equal the number of parameters
            of the `EllipseModel`

        matern_ellipse : EllipseModel
            EllipseModel to use for parameter estimation

        max_distance : float
            Maximum separation in distance unit that data will be fed
            into parameter fitting
            Units depend on physical_distance attribute (km if True, otherwise
            degrees).

        min_distance: float
            Minimum separation in distance unit that data
            will be fed into parameter fitting
            Units depend on physical_distance attribute (km if True, otherwise
            degrees).
            Note: Due to the way we compute the Matern function,
            it is undefined at dist == 0 even if the limit -> zero is obvious.

        delta_x_method="Modified_Met_Office": str
            How to compute distances between grid points
            For istropic variogram/covariances, this is a trivial problem;
            you can just take the haversine or
            Euclidean ("tunnel") distance as they are non-directional.

            But it is non trivial for anistropic cases,
            you have to define a set of orthogonal space. In HadSST4,
            Earth is assumed to be cylindrical "tin can" Earth,
            so you can just define the orthogonal space by
            lines of constant lat and lon (delta_x_method="Met_Office").

            The modified "Modified_Met_Office" is a variation to that,
            but allow the tin can get squished at the poles.
            (Sinusoidal projection). This does results in a problem:
            the zonal displacement now depends in which latitude
            you compute on (at the beginning latitude or at the end latitude).
            Here we take the average of the two.

        guesses=None: tuple of floats; None uses default guess values
            Initial guess values that get feeds in the optimizer for MLE.
            In scipy, you are required to do so (but R often doesn't).
            You should anyway; sometimes they do funny things
            if you don't (per recommendation of David Stephenson)

        bounds=None: tuple of floats; None uses default bounds values
            This is essentially a Bayesian "uniformative prior"
            that forces convergence if the optimizer hits the bound.
            For lower resolution fitting, this is rarely a problem.
            For higher resolution fits, this often interacts with
            the limit of the data you can put into the fit the optimizer
            may fail to converge if the input data is very smooth (aka ENSO
            region, where anomalies are smooth over very large (~10000km)
            scales).

        opt_method='Nelder-Mead': str
            scipy.optimize method. Nelder-Mead is the one used by [Karspeck]_.
            See https://docs.scipy.org/doc/scipy/tutorial/optimize.html
            for valid options

        tol=0.001: float
            Set convergence tolerance for scipy optimize.
            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize

            Note on new tol kwarg:
            For N-M, this sets the value to both xatol and fatol
            Default is 1E-4
            Since it affects accuracy of all values including rotation
            rotation angle 0.001 rad ~ 0.05 deg

        estimate_SE=None : str | None
            The code can estimate the standard error if the Matern parameters.
            This is not usually used or discussed for the purpose of kriging.
            Certain opt_method (gradient descent) can do this automatically
            using Fisher Info for certain covariance function,
            but is not possible for some nasty functions (aka Bessel
            func) gets involved nor it is possible for some optimisers
            (such as Nelder-Mead).
            The code does it using bootstrapping.

        n_jobs=DEFAULT_N_JOBS: int
            If parallel processing, number of threads to use.

        n_sim : int
            Number of simulations to bootstrap for SE estimation.

        physical_distance_selection : bool
            Select training data using physical distance (haversine distance) or
            Euclidean distance of degree difference between each position and
            all possible training data. Data that falls within the min and max
            distance values using the selected distance method are selected as
            training data.

        Returns
        -------
        params : xarray.Dataset
            Containing arrays for each parameter in the ellipse model class.
            Note that one array is likely to be "qc_code", which takes values:

                - 0: success
                - 2: success but with one parameter reaching upper
                  boundaries
                - 3: success with multiple parameters reaching the
                  boundaries (aka both Lx and Ly), can be both at lower or
                  upper boundaries
                - 9: fail, probably due to running out of maxiter (see
                  scipy.optimize.minimize kwargs "options)"
        """
        coords_dict = {
            "latitude": self.coords["latitude"].values,
            "longitude": self.coords["longitude"].values,
        }
        coords = xr.Coordinates(coords_dict)
        param_names = matern_ellipse.supercategory_params
        params = init_parameter_set(
            coords,
            parameters=param_names,
            default_value=default_value,
        )

        for mask_i, (grid_i, grid_j) in enumerate(
            zip(self.xi_masked, self.yi_masked)
        ):
            result = self.fit_ellipse_model(
                mask_i,
                matern_ellipse=matern_ellipse,
                max_distance=max_distance,
                min_distance=min_distance,
                delta_x_method=delta_x_method,
                guesses=guesses,
                bounds=bounds,
                opt_method=opt_method,
                tol=tol,
                estimate_SE=estimate_SE,
                n_jobs=n_jobs,
                n_sim=n_sim,
                physical_distance_selection=physical_distance_selection,
            )
            if result is None:
                continue

            for i, param_name in enumerate(param_names.keys()):
                params[param_name][grid_j, grid_i] = result["ModelParams"][i]

        return params

    def __str__(self):
        return str(self.__class__)
        # return str(self.__class__) + ": " + str(self.__dict__)


def _get_fit_score(model_params, bounds, niter) -> int:
    fit_success: int = 0
    for model_param, bb in zip(model_params, bounds):
        left_check = maths.isclose(model_param, bb[0], rel_tol=0.01)
        right_check = maths.isclose(model_param, bb[1], rel_tol=0.01)
        # left_advisory = (
        #     "near_left_bnd" if left_check else "not_near_left_bnd"
        # )
        # right_advisory = (
        #     "near_right_bnd" if right_check else "not_near_rgt_bnd"
        # )
        # print(
        #     "Convergence success after ",
        #     niter,
        #     " iterations: ",
        #     model_param,
        #     bb[0],
        #     bb[1],
        #     left_advisory,
        #     right_advisory,
        # )
        if left_check:
            fit_success = 1 if fit_success == 0 else 3
        if right_check:
            fit_success = 2 if fit_success == 0 else 3
    return fit_success


def init_parameter_set(
    coords: xr.Coordinates,
    parameters: dict[str, str],
    default_value: Any = np.nan,
) -> xr.Dataset:
    """
    Initialise the ellipse parameter dataset.

    Contains arrays for each of the parameters of the model.

    Parameters
    ----------
    coords : xarray.Coordinates
        The coordinate system of the output arrays. Note that this should match
        the coordinate system of the data used to fit the ellipse parameters.
    parameters
    default_value : Any
        Default value(s) to fill arrays where parameter estimation is not
        possible (typically due to masking). Typically, one should set a
        value that is appropriate to the type of the field. If a single
        value is provided, this is used for all fields.

    Returns
    -------
    params : xarray.Dataset
        With arrays described above.
    """
    if not is_iter(default_value):
        default_value = [default_value for _ in range(6)]
    if len(default_value) != len(parameters):
        raise ValueError("Cannot set 6 default values for input default values")
    params = xr.Dataset(coords=coords)
    # Define a 3D variable to hold the data
    for i, (param_name, unit) in enumerate(parameters.items()):
        params[param_name] = xr.DataArray(
            data=default_value[i],
            coords=coords,
            name=param_name,
            attrs={
                "units": unit,
            },
        )
    return params
