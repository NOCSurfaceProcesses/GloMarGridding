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
Functions and Classes for performing Kriging.

Interpolation using a Gaussian Process. Available methods are Simple and
Ordinary Kriging.
"""
################
# by A. Faulkner
# for python version >= 3.11
################

from abc import ABC, abstractmethod
from types import NoneType
from typing import Literal
from warnings import warn

import numpy as np

from glomar_gridding.utils import (
    adjust_small_negative,
    get_spatial_mean,
    intersect_mtlb,
)

KrigMethod = Literal["simple", "ordinary"]


class Kriging(ABC):
    """
    Class for Kriging.

    Do not use this class, use SimpleKriging or OrdinaryKriging classes.

    Parameters
    ----------
    covariance : numpy.ndarray
        The spatial covariance matrix. This can be a pre-computed matrix loaded
        into the environment, or computed from a Variogram class or using
        Ellipse methods.
    idx : numpy.ndarray
        The 1d indices of observation grid points. These values should be
        between 0 and (N * M) - 1 where N, M are the number of longitudes
        and latitudes respectively. Note that these values should also be
        computed using "C" ordering in numpy reshaping. They can be
        computed from a grid using glomar_gridding.grid.map_to_grid. Each
        value should only appear once. Points that contain more than 1
        observation should be averaged. Used to compute the Kriging weights.
    obs : numpy.ndarray
        The observation values. If there are multiple observations in any
        grid box then these values need to be averaged into one value per
        grid box.
    error_cov : numpy.ndarray | NoneType
        Optionally add error covariance values to the covariance between
        observation grid points.
    """

    def __init__(
        self,
        covariance: np.ndarray,
        idx: np.ndarray,
        obs: np.ndarray,
        error_cov: np.ndarray | NoneType = None,
    ) -> NoneType:
        if not hasattr(self, "method"):
            raise NotImplementedError(
                "Do not use the generic class directly, "
                + "use SimpleKriging or OrdinaryKriging"
            )

        # Must have at most 1 observation at each grid-idx
        self.n_obs = len(idx)
        if len(np.unique(idx)) != self.n_obs:
            raise ValueError(
                "Have repeated positions in `idx`. "
                + "Combine observations and re-run map_observations method."
            )

        self.n_grid_pts = covariance.shape[0]
        self.covariance = covariance
        self.idx = idx
        self.obs = obs
        self.error_cov = error_cov

        self.subset_error_covariance()

    def subset_error_covariance(self) -> NoneType:
        """
        Adjust the error covariance matrix to the grid size, check for nans or
        zeros on the diagonal.
        """
        if self.error_cov is not None:
            if self.error_cov.shape[0] != self.n_obs:
                self.error_cov = self.error_cov[
                    self.idx[:, None], self.idx[None, :]
                ]
            if (
                mismatch := np.logical_or(
                    np.isnan(self.error_cov.diagonal()),
                    self.error_cov.diagonal() == 0,
                )
            ).any():
                idx_keep = np.where(np.logical_not(mismatch))[0]
                drop_idx = self.idx[mismatch].tolist()
                msg = (
                    "Have nans or zeros on the error covariance diagonal. "
                    + "At positions "
                    + " ,".join(map(str, drop_idx))
                    + ". Filtering input accordingly"
                )
                warn(msg)
                self.idx = self.idx[idx_keep]
                self.obs = self.obs[idx_keep]
                self.error_cov = self.error_cov[
                    idx_keep[:, None], idx_keep[None, :]
                ]
                self.n_obs = len(self.idx)
            # Fill remaining NaNs with 0.0
            self.error_cov[np.isnan(self.error_cov)] = 0.0
        return None

    def set_kriging_weights(self, kriging_weights: np.ndarray) -> NoneType:
        """
        Set Kriging Weights.

        Sets the `kriging_weights` attribute.

        Parameters
        ----------
        kriging_weights : numpy.ndarray
            The pre-computed kriging_weights to use.
        """
        self.kriging_weights = kriging_weights
        return None

    @abstractmethod
    def get_kriging_weights(self) -> NoneType:
        r"""
        Compute the Kriging weights from the flattened grid indices where
        there is an observation. Optionally add an error covariance to the
        covariance between observation grid points.

        The Kriging weights are calculated as:

        .. math::
            (C_{obs} + E)^{-1} \times C_{cross}

        Where :math:`C_{obs}` is the spatial covariance between grid-points
        with observations, :math:`E` is the error covariance between grid-points
        with observations, and :math:`C_{cross}` is the covariance between
        grid-points with observations and all grid-points (including observation
        grid-points).

        Sets the `kriging_weights` attribute.
        """
        raise NotImplementedError(
            "`get_kriging_weights` not implemented for default class"
        )

    @abstractmethod
    def kriging_weights_from_inverse(
        self,
        inv: np.ndarray,
    ) -> NoneType:
        r"""
        Compute the Kriging weights from the flattened grid indices where
        there is an observation, using a pre-computed inverse of the covariance
        between grid-points with observations.

        The Kriging weights are calculated as:

        .. math::
            (C_{obs} + E)^{-1} \times C_{cross}

        Where :math:`C_{obs}` is the spatial covariance between grid-points
        with observations, :math:`E` is the error covariance between grid-points
        with observations, and :math:`C_{cross}` is the covariance between
        grid-points with observations and all grid-points (including observation
        grid-points).

        Sets the `kriging_weights` attribute.

        Parameters
        ----------
        inv : numpy.ndarray
            The pre-computed inverse of the covariance between grid-points with
            observations. :math:`(C_{obs} + E)^{-1}`
        """
        raise NotImplementedError(
            "`kriging_weights_from_inverse` not implemented for default class"
        )

    @abstractmethod
    def solve(self) -> np.ndarray:
        r"""
        Solves the Kriging problem. Computes the Kriging weights if the
        `kriging_weights` attribute is not already set. The solution to Kriging
        is:

        .. math::
            (C_{obs} + E)^{-1} \times C_{cross} \times y

        Where :math:`C_{obs}` is the spatial covariance between grid-points
        with observations, :math:`E` is the error covariance between grid-points
        with observations, :math:`C_{cross}` is the covariance between
        grid-points with observations and all grid-points (including observation
        grid-points), and :math:`y` are the observation values.

        Returns
        -------
        numpy.ndarray
            The solution to the Kriging problem (as a Vector, this may need to
            be re-shaped appropriately as a post-processing step).
        """
        raise NotImplementedError("`solve` not implemented for default class")

    @abstractmethod
    def get_uncertainty(self, full_covariance: bool = False) -> np.ndarray:
        """
        Compute the kriging uncertainty. This requires the attribute
        `kriging_weights` to be computed.

        Parameters
        ----------
        full_covariance : bool
            Return the full Kriging covariance if True. Defaults to False.

        Returns
        -------
        uncert : numpy.ndarray
            The Kriging uncertainty or the full Kriging covariance.
        """
        raise NotImplementedError(
            "`get_uncertainty` not implemented for default class"
        )

    @abstractmethod
    def constraint_mask(self) -> np.ndarray:
        r"""
        Compute the observational constraint mask (A14 in [Morice_2021]_) to
        determine if a grid point should be masked/weights modified by how far
        it is to its near observed point

        Note: typo in Section A4 in [Morice_2021]_ (confirmed by authors).

        Equation to use is A14 is incorrect. Easily noticeable because
        dimensionally incorrect is wrong, but the correct answer is easy to
        figure out.

        Correct Equation (extra matrix inverse for :math:`C_{obs} + E`):

        .. math::
            1 - \frac{
                diag(C - C_{cross}^T \times (C_{obs} + E)^{-1}
                         \times C_{cross})
            }{diag(C)} < \alpha

        This can be re-written as:

        .. math::
            \frac{
                diag(C_{cross}^T \times (C_{obs} + E)^{-1} \times C_{cross})
            }{diag(C)} < \alpha

        :math:`\alpha` is chosen to be 0.25 in the UKMO paper

        Written by S. Chan, modified by J. Siddons.

        This requires that the `kriging_weights` attribute is set.

        Returns
        -------
        constraint_mask : numpy.ndarray
            Constraint mask values, the left-hand-side of equation A14 from
            Morice et al. (2021). This is a vector of length `k_obs.size[0]`.

        References
        ----------
        [Morice_2021]_: https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2019JD032361
        """
        raise NotImplementedError(
            "`constraint_mask` not implemented for default class"
        )


class SimpleKriging(Kriging):
    r"""
    Class for SimpleKriging.

    The equation for simple Kriging is:

    .. math::
        (C_{obs} + E)^{-1} \times C_{cross} \times y + \mu

    Where :math:`\mu` is a constant known mean, typically this is 0.

    Parameters
    ----------
    covariance : numpy.ndarray
        The spatial covariance matrix. This can be a pre-computed matrix loaded
        into the environment, or computed from a Variogram class or using
        Ellipse methods.
    idx : numpy.ndarray[int] | list[int]
        The 1d indices of observation grid points. These values should be
        between 0 and (N * M) - 1 where N, M are the number of longitudes
        and latitudes respectively. Note that these values should also be
        computed using "C" ordering in numpy reshaping. They can be
        computed from a grid using glomar_gridding.grid.map_to_grid. Each
        value should only appear once. Points that contain more than 1
        observation should be averaged
    obs : numpy.ndarray[float]
        The observation values. If there are multiple observations in any
        grid box then these values need to be averaged into one value per
        grid box.
    error_cov : numpy.ndarray | NoneType
        Optionally add error covariance values to the covariance between
        observation grid points.
    """

    method: str = "simple"

    def get_kriging_weights(self) -> NoneType:
        r"""
        Compute the Kriging weights from the flattened grid indices where
        there is an observation. Optionally add an error covariance to the
        covariance between observation grid points.

        The Kriging weights are calculated as:

        .. math::
            (C_{obs} + E)^{-1} \times C_{cross}

        Where :math:`C_{obs}` is the spatial covariance between grid-points
        with observations, :math:`E` is the error covariance between grid-points
        with observations, and :math:`C_{cross}` is the covariance between
        grid-points with observations and all grid-points (including observation
        grid-points).

        Sets the `kriging_weights` attribute.
        """
        obs_obs_cov = self.covariance[
            self.idx[:, None], self.idx[None, :]
        ].copy()
        obs_grid_cov = self.covariance[self.idx, :]

        # Add error covariance
        if self.error_cov is not None:
            obs_obs_cov += self.error_cov
        self.kriging_weights = np.linalg.solve(obs_obs_cov, obs_grid_cov).T

        return None

    def kriging_weights_from_inverse(
        self,
        inv: np.ndarray,
    ) -> NoneType:
        r"""
        Compute the Kriging weights from the flattened grid indices where
        there is an observation, using a pre-computed inverse of the covariance
        between grid-points with observations.

        The Kriging weights are calculated as:

        .. math::
            (C_{obs} + E)^{-1} \times C_{cross}

        Where :math:`C_{obs}` is the spatial covariance between grid-points
        with observations, :math:`E` is the error covariance between grid-points
        with observations, and :math:`C_{cross}` is the covariance between
        grid-points with observations and all grid-points (including observation
        grid-points).

        Sets the `kriging_weights` attribute.

        Parameters
        ----------
        inv : numpy.ndarray
            The pre-computed inverse of the covariance between grid-points with
            observations. :math:`(C_{obs} + E)^{-1}`
        """
        if self.n_obs != inv.shape[0]:
            raise ValueError("inv must be square with side length == len(idx)")
        obs_grid_cov = self.covariance[self.idx, :]
        self.kriging_weights = (inv @ obs_grid_cov).T

    def solve(
        self,
        mean: np.ndarray | float = 0.0,
    ) -> np.ndarray:
        r"""
        Solves the simple Kriging problem. Computes the Kriging weights if the
        `kriging_weights` attribute is not already set. The solution to Kriging
        is:

        .. math::
            (C_{obs} + E)^{-1} \times C_{cross} \times y

        Where :math:`C_{obs}` is the spatial covariance between grid-points
        with observations, :math:`E` is the error covariance between grid-points
        with observations, :math:`C_{cross}` is the covariance between
        grid-points with observations and all grid-points (including observation
        grid-points), and :math:`y` are the observation values.

        Parameters
        ----------
        mean : numpy.ndarray | float
            Constant, known, mean value of the system. Defaults to 0.0.

        Returns
        -------
        numpy.ndarray
            The solution to the simple Kriging problem (as a Vector, this may
            need to be re-shaped appropriately as a post-processing step).

        Examples
        --------
        >>> SK = SimpleKriging(interp_covariance)
        >>> SK.solve(obs, idx, error_covariance)
        """
        if not hasattr(self, "kriging_weights"):
            self.get_kriging_weights()

        return self.kriging_weights @ self.obs + mean

    def get_uncertainty(self, full_covariance: bool = False) -> np.ndarray:
        """
        Compute the kriging uncertainty. This requires the attribute
        `kriging_weights` to be computed.

        Parameters
        ----------
        full_covariance : bool
            Return the full Kriging covariance if True. Defaults to False.

        Returns
        -------
        uncert : numpy.ndarray
            The Kriging uncertainty or the full Kriging covariance.
        """
        if not hasattr(self, "kriging_weights"):
            raise KeyError("Please compute Kriging Weights first")

        obs_grid_cov = self.covariance[self.idx, :]

        kriging_weights = self.kriging_weights @ obs_grid_cov

        if full_covariance:
            return self.covariance - kriging_weights

        dz_squared = np.diag(self.covariance - kriging_weights)
        dz_squared = adjust_small_negative(dz_squared)
        uncert = np.sqrt(dz_squared)
        uncert[np.isnan(uncert)] = 0.0
        return uncert

    def constraint_mask(self) -> np.ndarray:
        r"""
        Compute the observational constraint mask (A14 in [Morice_2021]_) to
        determine if a grid point should be masked/weights modified by how far
        it is to its near observed point

        Note: typo in Section A4 in [Morice_2021]_ (confirmed by authors).

        Equation to use is A14 is incorrect. Easily noticeable because
        dimensionally incorrect is wrong, but the correct answer is easy to
        figure out.

        Correct Equation (extra matrix inverse for :math:`C_{obs} + E`):

        .. math::
            1 - \frac{
                diag(C - C_{cross}^T \times (C_{obs} + E)^{-1}
                         \times C_{cross})
            }{diag(C)} < \alpha

        This can be re-written as:

        .. math::
            \frac{
                diag(C_{cross}^T \times (C_{obs} + E)^{-1} \times C_{cross})
            }{diag(C)} < \alpha

        :math:`\alpha` is chosen to be 0.25 in the UKMO paper

        Written by S. Chan, modified by J. Siddons.

        This requires that the `kriging_weights` attribute is set.

        Returns
        -------
        constraint_mask : numpy.ndarray
            Constraint mask values, the left-hand-side of equation A14 from
            Morice et al. (2021). This is a vector of length `k_obs.size[0]`.

        References
        ----------
        [Morice_2021]_: https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2019JD032361
        """
        if not hasattr(self, "kriging_weights"):
            raise KeyError("Please compute Kriging Weights first")

        numerator = np.diag(
            self.covariance[:, self.idx] @ self.kriging_weights.T
        )
        denominator = np.diag(self.covariance)
        return np.divide(numerator, denominator)


class OrdinaryKriging(Kriging):
    r"""
    Class for OrdinaryKriging.

    The equation for ordinary Kriging is:

    .. math::
        (C_{obs} + E)^{-1} \times C_{cross} \times y

    with a constant but unknown mean.

    In this case, the :math:`C_{obs}`, :math:`C_{cross}` and :math:`y` values
    are extended with a Lagrange multiplier term, ensuring that the Kriging
    weights are constrained to sum to 1.

    The matrix :math:`C_{obs}` is extended by one row and one column, each
    containing the value 1, except at the diagonal point, which is 0. The
    :math:`C_{cross}` matrix is extended by an extra row containing values of 1.
    Finally, the grid observations :math:`y` is extended by a single value of 0
    at the end of the vector.

    Parameters
    ----------
    covariance : numpy.ndarray
        The spatial covariance matrix. This can be a pre-computed matrix loaded
        into the environment, or computed from a Variogram class or using
        Ellipse methods.
    idx : numpy.ndarray
        The 1d indices of observation grid points. These values should be
        between 0 and (N * M) - 1 where N, M are the number of longitudes
        and latitudes respectively. Note that these values should also be
        computed using "C" ordering in numpy reshaping. They can be
        computed from a grid using glomar_gridding.grid.map_to_grid. Each
        value should only appear once. Points that contain more than 1
        observation should be averaged. Used to compute the Kriging weights.
    obs : numpy.ndarray
        The observation values. If there are multiple observations in any
        grid box then these values need to be averaged into one value per
        grid box.
    error_cov : numpy.ndarray | NoneType
        Optionally add error covariance values to the covariance between
        observation grid points.
    """

    method: str = "ordinary"

    def get_kriging_weights(self) -> NoneType:
        r"""
        Compute the Kriging weights from the flattened grid indices where
        there is an observation. Optionally add an error covariance to the
        covariance between observation grid points.

        The Kriging weights are calculated as:

        .. math::
            (C_{obs} + E)^{-1} \times C_{cross}

        Where :math:`C_{obs}` is the spatial covariance between grid-points
        with observations, :math:`E` is the error covariance between grid-points
        with observations, and :math:`C_{cross}` is the covariance between
        grid-points with observations and all grid-points (including observation
        grid-points).

        In this case, the :math:`C_{obs}`, :math:`C_{cross}` and are extended
        with a Lagrange multiplier term, ensuring that the Kriging weights are
        constrained to sum to 1.

        The matrix :math:`C_{obs}` is extended by one row and one column, each
        containing the value 1, except at the diagonal point, which is 0. The
        :math:`C_{cross}` matrix is extended by an extra row containing values
        of 1.

        Sets the `kriging_weights` attribute.
        """
        obs_obs_cov = self.covariance[
            self.idx[:, None], self.idx[None, :]
        ].copy()
        obs_grid_cov = self.covariance[self.idx, :]

        # Add error covariance
        if self.error_cov is not None:
            obs_obs_cov += self.error_cov

        # Add Lagrange multiplier
        ones_n = np.ones((1, self.n_obs), dtype=self.covariance.dtype)
        ones_m = np.ones((1, self.n_grid_pts), dtype=self.covariance.dtype)
        zero = np.zeros((1, 1), dtype=self.covariance.dtype)
        obs_obs_cov = np.block([[obs_obs_cov, ones_n.T], [ones_n, zero]])
        obs_grid_cov = np.concatenate((obs_grid_cov, ones_m), axis=0)
        self.kriging_weights = (
            np.linalg.solve(obs_obs_cov, obs_grid_cov).T
        ).astype(self.covariance.dtype)

        return None

    def kriging_weights_from_inverse(
        self,
        inv: np.ndarray,
    ) -> NoneType:
        r"""
        Compute the Kriging weights from the flattened grid indices where
        there is an observation, using a pre-computed inverse of the covariance
        between grid-points with observations.

        The Kriging weights are calculated as:

        .. math::
            (C_{obs} + E)^{-1} \times C_{cross}

        Where :math:`C_{obs}` is the spatial covariance between grid-points
        with observations, :math:`E` is the error covariance between grid-points
        with observations, and :math:`C_{cross}` is the covariance between
        grid-points with observations and all grid-points (including observation
        grid-points).

        In this case, the inverse matrix must be computed from the covariance
        between observation grid-points with the Lagrange multiplier applied.

        This method is appropriate if one wants to compute the constraint mask
        which requires simple Kriging weights, which can be computed from the
        unextended covariance inverse. The extended inverse can then be
        calculated from that inverse.

        Sets the `kriging_weights` attribute.

        Parameters
        ----------
        inv : numpy.ndarray
            The pre-computed inverse of the covariance between grid-points with
            observations. :math:`(C_{obs} + E)^{-1}`
        """
        if self.n_obs != inv.shape[0] - 1:
            raise ValueError("inv must be square with side length == len(idx)")
        obs_grid_cov = self.covariance[self.idx, :]

        # Add Lagrange multiplier
        ones_m = np.ones((1, self.n_grid_pts), dtype=self.covariance.dtype)
        obs_grid_cov = np.concatenate((obs_grid_cov, ones_m), axis=0)
        self.kriging_weights = ((inv @ obs_grid_cov).T).astype(
            self.covariance.dtype
        )

    def solve(self) -> np.ndarray:
        r"""
        Solves the ordinary Kriging problem. Computes the Kriging weights if the
        `kriging_weights` attribute is not already set. The solution to Kriging
        is:

        .. math::
            (C_{obs} + E)^{-1} \times C_{cross} \times y

        Where :math:`C_{obs}` is the spatial covariance between grid-points
        with observations, :math:`E` is the error covariance between grid-points
        with observations, :math:`C_{cross}` is the covariance between
        grid-points with observations and all grid-points (including observation
        grid-points), and :math:`y` are the observation values.

        In this case, the :math:`C_{obs}`, :math:`C_{cross}` and are extended
        with a Lagrange multiplier term, ensuring that the Kriging weights are
        constrained to sum to 1.

        The matrix :math:`C_{obs}` is extended by one row and one column, each
        containing the value 1, except at the diagonal point, which is 0. The
        :math:`C_{cross}` matrix is extended by an extra row containing values
        of 1.

        Returns
        -------
        numpy.ndarray
            The solution to the ordinary Kriging problem (as a Vector, this may
            need to be re-shaped appropriately as a post-processing step).

        Examples
        --------
        >>> OK = OrdinaryKriging(interp_covariance)
        >>> OK.solve(obs, idx, error_covariance)
        """
        if not hasattr(self, "kriging_weights"):
            self.get_kriging_weights()

        # Add Lagrange multiplier
        grid_obs = np.append(self.obs, 0).astype(self.obs.dtype)

        return self.kriging_weights @ grid_obs

    def get_uncertainty(self, full_covariance: bool = False) -> np.ndarray:
        """
        Compute the kriging uncertainty. This requires the attribute
        `kriging_weights` to be computed.

        Parameters
        ----------
        full_covariance : bool
            Return the full Kriging covariance if True. Defaults to False.

        Returns
        -------
        uncert : numpy.ndarray
            The Kriging uncertainty or the full Kriging covariance.
        """
        if not hasattr(self, "kriging_weights"):
            raise KeyError("Please compute Kriging Weights first")

        M = self.covariance.shape[0]
        obs_grid_cov = self.covariance[self.idx, :].copy()
        ones_m = np.ones((1, M), dtype=self.covariance.dtype)
        obs_grid_cov = np.concatenate((obs_grid_cov, ones_m), axis=0)
        kriging_weights = self.kriging_weights @ obs_grid_cov

        # NOTE: self.kriging_weights includes the Lagrange multiplier
        # (alpha) as its final element, and obs_grid_cov has been
        # extended by a row of ones. The product therefore already
        # equals W @ C_cross + alpha 1^T, so the ordinary Kriging
        # error covariance C - W @ C_cross - alpha 1^T is obtained
        # without any further alpha term.

        if full_covariance:
            return self.covariance - kriging_weights

        uncert_squared = np.diag(self.covariance - kriging_weights)
        uncert_squared = adjust_small_negative(uncert_squared)
        uncert = np.sqrt(uncert_squared)
        uncert[np.isnan(uncert)] = 0.0

        return uncert

    def constraint_mask(
        self,
        simple_kriging_weights: np.ndarray | NoneType = None,
    ) -> np.ndarray:
        r"""
        Compute the observational constraint mask (A14 in [Morice_2021]_) to
        determine if a grid point should be masked/weights modified by how far
        it is to its near observed point

        Note: typo in Section A4 in [Morice_2021]_ (confirmed by authors).

        Equation to use is A14 is incorrect. Easily noticeable because
        dimensionally incorrect is wrong, but the correct answer is easy to
        figure out.

        Correct Equation (extra matrix inverse for :math:`C_{obs} + E`):

        .. math::
            1 - \frac{
                diag(C - C_{cross}^T \times (C_{obs} + E)^{-1}
                         \times C_{cross})
            }{diag(C)} < \alpha

        This can be re-written as:

        .. math::
            \frac{
                diag(C_{cross}^T \times (C_{obs} + E)^{-1} \times C_{cross})
            }{diag(C)} < \alpha

        :math:`\alpha` is chosen to be 0.25 in the UKMO paper

        Written by S. Chan, modified by J. Siddons.

        This requires the Kriging weights from simple Kriging. If these are
        not provided as an input, then they are calculated.

        Parameters
        ----------
        simple_kriging_weights : numpy.ndarray | NoneType,
            The Kriging weights for the equivalent simple Kriging system.
            error covariance.

        Returns
        -------
        constraint_mask : numpy.ndarray
            Constraint mask values, the left-hand-side of equation A14 from
            Morice et al. (2021). This is a vector of length `k_obs.size[0]`.

        References
        ----------
        [Morice_2021]_: https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2019JD032361
        """
        if simple_kriging_weights is None:
            obs_obs_cov = self.covariance[
                self.idx[:, None], self.idx[None, :]
            ].copy()
            obs_grid_cov = self.covariance[self.idx, :]

            # Add error covariance
            if self.error_cov is not None:
                obs_obs_cov += self.error_cov
            simple_kriging_weights = np.linalg.solve(
                obs_obs_cov, obs_grid_cov
            ).T

        numerator = np.diag(
            self.covariance[:, self.idx] @ simple_kriging_weights.T
        )
        denominator = np.diag(self.covariance)
        return np.divide(numerator, denominator)

    def extended_inverse(self, simple_inv: np.ndarray) -> np.ndarray:
        r"""
        Compute the inverse of a covariance matrix :math:`S = C_{obs} + E`, and
        use that to compute the inverse of the extended version of the
        covariance matrix with Lagrange multipliers, used by Ordinary Kriging.

        This is useful when one needs to perform BOTH simple and ordinary
        Kriging, or when one wishes to compute the constraint mask for
        ordinary Kriging which requires the Kriging weights for the equivalent
        simple Kriging problem.

        The extended form of S is given by:

        .. math::
            \begin{pmatrix}
            &   & & 1 \\
            & S & & \vdots \\
            &   & & 1 \\
            1 & \dots & 1 & 0 \\
            \end{pmatrix}

        This approach follows Guttman 1946 10.1214/aoms/1177730946

        Parameters
        ----------
        simple_inv : numpy.matrix
            Inverse of the covariance between observation grid-points

        Returns
        -------
        numpy.matrix
            Inverse of the extended covariance matrix between observation
            grid-points including the Lagrange multiplier factors.
        """
        return _extended_inverse(simple_inv).astype(self.covariance.dtype)


def _extended_inverse(simple_inv: np.ndarray) -> np.ndarray:
    if len(simple_inv.shape) != 2:
        raise ValueError("S must be a matrix")

    d = 0
    B = np.ones((simple_inv.shape[0], 1), dtype=simple_inv.dtype)

    E = np.matmul(simple_inv, B)
    f = d - np.matmul(B.T, E)
    finv = 1 / f
    G = finv * E.T
    # H = finv * np.matmul(B.T, Ainv)
    K = simple_inv + np.matmul(E, G)

    return np.block([[K, -G.T], [-G, finv]])


def prep_obs_for_kriging(
    unmask_idx: np.ndarray,
    unique_obs_idx: np.ndarray,
    weights: np.ndarray,
    obs: np.ndarray,
    remove_obs_mean: int = 0,
    obs_bias: np.ndarray | NoneType = None,
    error_cov: np.ndarray | NoneType = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prep masked observations for Kriging. Combines observations in the same
    grid box to a single averaged observation using a weighted average.

    Parameters
    ----------
    unmask_idx : np.ndarray[int]
        Indices of all un-masked points for chosen date.
    unique_obs_idx : np.ndarray[int]
        Unique indices of all measurement points for a chosen date,
        representative of the indices of gridboxes, which have => 1 measurement.
    weights : np.ndarray[float]
        Weight matrix (inverse of counts of observations).
    obs : np.ndarray[float]
        All point observations/measurements for the chosen date.
    remove_obs_mean: int
        Should the mean or median from obs be removed and added back onto obs?

            - 0 = No (default action)
            - 1 = the mean is removed
            - 2 = the median is removed
            - 3 = the spatial meam os removed

        Note that the mean will need to be reapplied to the Kriging result.
    obs_bias : np.ndarray[float] | NoneType
        Bias of all measurement points for a chosen date (corresponds to x_obs).

    Returns
    -------
    obs_idx : numpy.ndarray[int]
        Subset of grid-box indices containing observations that are unmasked.
    grid_obs : numpy.ndarray[float]
        Unmasked and combined observations
    """
    obs_idx = get_unmasked_obs_indices(unmask_idx, unique_obs_idx)

    if obs_bias is not None:
        print("With bias")
        grid_obs = weights @ (obs - obs_bias)
    else:
        grid_obs = weights @ obs

    grid_obs = np.squeeze(grid_obs) if len(grid_obs) > 1 else grid_obs

    match remove_obs_mean:
        case 0:
            grid_obs_av = None
        case 1:
            grid_obs_av = np.ma.average(grid_obs)
            grid_obs = grid_obs - grid_obs_av
        case 2:
            grid_obs_av = np.ma.median(grid_obs)
            grid_obs = grid_obs - grid_obs_av
        case 3:
            if error_cov is None:
                raise ValueError(
                    "'remove_obs_mean = 3 requires error covariance"
                )
            grid_obs_av = get_spatial_mean(grid_obs, error_cov)
            grid_obs = grid_obs - grid_obs_av
        case _:
            raise ValueError("Unknown 'remove_obs_mean' value")

    return obs_idx, grid_obs


def get_unmasked_obs_indices(
    unmask_idx: np.ndarray,
    unique_obs_idx: np.ndarray,
) -> np.ndarray:
    """
    Get grid indices with observations from un-masked grid-box indices and
    unique grid-box indices with observations.

    Parameters
    ----------
    unmask_idx : np.ndarray[int]
        List of all unmasked grid-box indices.
    unique_obs_idx : np.ndarray[int]
        Indices of grid-boxes with observations.

    Returns
    -------
    obs_idx : np.ndarray[int]
        Subset of grid-box indices containing observations that are unmasked.
    """
    unmask_idx = np.squeeze(unmask_idx) if unmask_idx.ndim > 1 else unmask_idx
    _, obs_idx, _ = intersect_mtlb(unmask_idx, unique_obs_idx)
    # index of the sorted unique (iid) in the full iid array
    obs_idx = obs_idx.astype(int)

    return obs_idx


def kriging_simple(
    obs_obs_cov: np.ndarray,
    obs_grid_cov: np.ndarray,
    grid_obs: np.ndarray,
    interp_cov: np.ndarray,
    mean: float | np.ndarray = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform Simple Kriging assuming a constant known mean.

    This function is deprecated in favour of SimpleKriging class. It will be
    removed in version 1.0.0.

    Parameters
    ----------
    obs_obs_cov : np.ndarray[float]
        Covariance between all measured grid points plus the
        covariance due to measurements (i.e. measurement noise, bias noise, and
        sampling noise). Can include error covariance terms.
    obs_grid_cov : np.ndarray[float]
        Covariance between the all (predicted) grid points and measured points.
        Does not contain error covarance.
    grid_obs : np.ndarray[float]
        Gridded measurements (all measurement points averaged onto the output
        gridboxes).
    interp_cov : np.ndarray[float]
        interpolation covariance of all output grid points (each point in time
        and all points against each other).
    mean : float
        The constant mean of the output field.

    Returns
    -------
    z_obs : np.ndarray[float]
        Full set of values for the whole domain derived from the observation
        points using simple kriging.
    dz : np.ndarray[float]
        Uncertainty associated with the simple kriging.
    """
    warn(
        "kriging_simple is deprecated and will be removed in version v1.0.0, "
        + "use SimpleKriging",
        DeprecationWarning,
    )
    kriging_weights = np.linalg.solve(obs_obs_cov, obs_grid_cov).T
    kriged_result = kriging_weights @ grid_obs

    kriging_weights = kriging_weights @ obs_grid_cov
    dz_squared = np.diag(interp_cov - kriging_weights)
    dz_squared = adjust_small_negative(dz_squared)
    uncert = np.sqrt(dz_squared)
    uncert[np.isnan(uncert)] = 0.0

    print("Simple Kriging Complete")
    return kriged_result + mean, uncert


def kriging_ordinary(
    obs_obs_cov: np.ndarray,
    obs_grid_cov: np.ndarray,
    grid_obs: np.ndarray,
    interp_cov: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform Ordinary Kriging with unknown but constant mean.

    This function is deprecated in favour of OrdinaryKriging class. It will be
    removed in version 1.0.0.

    Parameters
    ----------
    obs_obs_cov : np.ndarray[float]
        Covariance between all measured grid points plus the covariance due to
        measurements (i.e. measurement noise, bias noise, and sampling noise).
        Can include error covariance terms, if these are being used.
    obs_grid_cov : np.ndarray[float]
        Covariance between the all (predicted) grid points and measured points.
        Does not contain error covarance.
    grid_obs : np.ndarray[float]
        Gridded measurements (all measurement points averaged onto the output
        gridboxes).
    interp_cov : np.ndarray[float]
        Interpolation covariance of all output grid points (each point in time
        and all points against each other).

    Returns
    -------
    z_obs : np.ndarray[float]
        Full set of values for the whole domain derived from the observation
        points using ordinary kriging.
    dz : np.ndarray[float]
        Uncertainty associated with the ordinary kriging.
    """
    warn(
        "kriging_ordinary is deprecated and will be removed in version v1.0.0, "
        + "use OrdinaryKriging",
        DeprecationWarning,
    )
    # Convert to ordinary kriging, add Lagrangian multiplier
    N, M = obs_grid_cov.shape
    obs_obs_cov = np.block(
        [[obs_obs_cov, np.ones((N, 1))], [np.ones((1, N)), 0]]
    )
    obs_grid_cov = np.concatenate((obs_grid_cov, np.ones((1, M))), axis=0)
    grid_obs = np.append(grid_obs, 0)

    # kriging_weights includes the Lagrange multiplier as its last elemen, so
    # the product already contains the multiplier term.

    kriging_weights = np.linalg.solve(obs_obs_cov, obs_grid_cov).T
    kriged_result = kriging_weights @ grid_obs

    kriging_weights = kriging_weights @ obs_grid_cov
    uncert_squared = np.diag(interp_cov - kriging_weights)
    uncert_squared = adjust_small_negative(uncert_squared)
    uncert = np.sqrt(uncert_squared)
    uncert[np.isnan(uncert)] = 0.0

    print("Ordinary Kriging Complete")
    return kriged_result, uncert


def constraint_mask(
    obs_obs_cov: np.ndarray,
    obs_grid_cov: np.ndarray,
    interp_cov: np.ndarray,
) -> np.ndarray:
    r"""
    Compute the observational constraint mask (A14 in [Morice_2021]_) to
    determine if a grid point should be masked/weights modified by how far it is
    to its near observed point.

    Note: typo in Section A4 in Morice et al 2021 (confirmed by authors).

    Equation to use is A14 is incorrect. Easily noticeable because dimensionally
    incorrect is wrong, but the correct answer is easy to figure out.

    Correct Equation (extra matrix inverse for C+R):

    .. math::
        1 - diag(C(X*,X*) - k*^T \times (C+R)^{-1} \times k*)  / diag(C(X*,X*))
        < \alpha

    This can be re-written as:

    .. math::
        diag(k*^T \times (C+R)^{-1} \times k*) / diag(C(X*, X*)) < \alpha

    :math:`\alpha` is chosen to be 0.25 in the UKMO paper

    Written by S. Chan, modified by J. Siddons.

    Parameters
    ----------
    obs_obs_cov : np.ndarray[float]
        Covariance between all measured grid points plus the covariance due to
        measurements (i.e. measurement noise, bias noise, and sampling noise).
        Can include error covariance terms, if these are being used. This is
        `C + R` in the above equation.
    obs_grid_cov : np.ndarray[float]
        Covariance between the all (predicted) grid points and measured points.
        Does not contain error covarance. This is `k*` in the above equation.
    interp_cov : np.ndarray[float]
        Interpolation covariance of all output grid points (each point in time
        and all points against each other). This is `C(X*, X*)` in the above
        equation.

    Returns
    -------
    constraint_mask : numpy.ndarray
        Constraint mask values, the left-hand-side of equation A14 from Morice
        et al. (2021). This is a vector of length `k_obs.size[0]`.

    References
    ----------
    Morice et al. (2021) [Morice_2021]_
    """
    # ky_inv = np.linalg.inv(k_obs + err_cov)
    # NOTE: Ax = b => x = A^{-1}b (x = solve(A, b))
    Kinv_kstar = np.linalg.solve(obs_obs_cov, obs_grid_cov)
    numerator = np.diag(obs_grid_cov.T @ Kinv_kstar)
    denominator = np.diag(interp_cov)
    constraint_mask = numerator / denominator
    # constraint_mask has the length of number of grid points
    # (obs-covered and interpolated.)
    return constraint_mask
