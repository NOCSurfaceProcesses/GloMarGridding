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

"""Functions for creating grids and mapping observations to a grid"""

from collections.abc import Callable, Iterable
from types import NoneType
from typing import Any, Literal

import numpy as np
import polars as pl
import xarray as xr

from glomar_gridding.variogram import (
    ExponentialVariogram,
    GaussianVariogram,
    MaternVariogram,
    SphericalVariogram,
    variogram_to_covariance,
)

from .distances import calculate_distance_matrix, haversine_distance_from_frame
from .kriging import (
    SimpleKriging,
    OrdinaryKriging,
)
from .stochastic import StochasticKriging
from .utils import filter_bounds, find_nearest, select_bounds


def map_to_grid(
    obs: pl.DataFrame,
    grid: xr.DataArray,
    obs_coords: list[str] = ["lat", "lon"],
    grid_coords: list[str] = ["latitude", "longitude"],
    sort: bool = True,
    bounds: list[tuple[float, float]] | None = None,
    add_grid_pts: bool = True,
    grid_prefix: str = "grid_",
) -> pl.DataFrame:
    """
    Align an observation dataframe to a grid defined by an xarray DataArray.

    Maps observations to the nearest grid-point, and sorts the data by the
    1d index of the DataArray in a row-major format.

    The grid defined by the latitude and longitude coordinates of the input
    DataArray is then used as the output grid of the Gridding process.

    Parameters
    ----------
    obs : polars.DataFrame
        The observational DataFrame containing positional data with latitude,
        longitude values within the `obs_latname` and `obs_lonname` columns
        respectively. Observations are mapped to the nearest grid-point in the
        grid.
    grid : xarray.DataArray
        Contains the grid coordinates to map observations to.
    obs_coords : list[str]
        Names of the column containing positional values in the input
        observational DataFrame.
    grid_coords : list[str]
        Names of the coordinates in the input grid DataArray used to define the
        grid.
    sort : bool
        Sort the observational DataFrame by the grid index
    bounds : list[tuple[float, float]] | None
        Optionally filter the grid and DataFrame to fall within spatial bounds.
        This list must have the same size and ordering as `obs_coords` and
        `grid_coords` arguments.
    add_grid_pts : bool
        Add the grid positional information to the observational DataFrame.
    grid_prefix : str
        Prefix to use for the new grid columns in the observational DataFrame.

    Returns
    -------
    obs : pandas.DataFrame
        Containing additional `grid_*`, and `grid_idx` values
        indicating the positions and grid index of the observation
        respectively. The DataFrame is also sorted (ascendingly) by the
        `grid_idx` columns for consistency with the gridding functions.

    Examples
    --------
    >>> obs = pl.read_csv("/path/to/obs.csv")
    >>> grid = grid_from_resolution(
            resolution=5,
            bounds=[(-87.5, 90), (-177.5, 180)],  # Lower bound is centre
            coord_names=["lat", "lon"]
        )
    >>> obs = map_to_grid(obs, grid, grid_coords=["lat", "lon"])
    """
    if bounds is not None:
        grid = select_bounds(grid, bounds, grid_coords)
        obs = filter_bounds(obs, bounds, obs_coords)

    grid_size = grid.shape

    grid_idx: list[list[int]] = []
    obs_to_grid_pos: list[np.ndarray] = []
    for grid_coord, obs_coord in zip(grid_coords, obs_coords):
        grid_pos = grid.coords[grid_coord].values
        _grid_idx, _obs_to_grid_pos = find_nearest(
            grid_pos, obs[obs_coord].to_numpy()
        )
        grid_idx.append(_grid_idx)
        obs_to_grid_pos.append(_obs_to_grid_pos)
        del _grid_idx, _obs_to_grid_pos

    flattened_idx = np.ravel_multi_index(
        grid_idx,
        grid_size,
        order="C",  # row-major
    )

    grid_idx_col = grid_prefix + "idx"
    obs = obs.with_columns(pl.Series(grid_idx_col, flattened_idx))
    if add_grid_pts:
        obs = obs.with_columns(
            [
                pl.Series(grid_prefix + obs_coord, grid_pos)
                for grid_pos, obs_coord in zip(obs_to_grid_pos, obs_coords)
            ]
        )

    if sort:
        obs = obs.sort(grid_idx_col, descending=False)

    return obs


def grid_from_resolution(
    resolution: float | list[float],
    bounds: list[tuple[float, float]],
    coord_names: list[str],
    definition: Literal["center", "left"] = "center",
) -> xr.DataArray:
    """
    Generate a grid from a resolution value, or a list of resolutions for
    given boundaries and coordinate names.

    Note that all list inputs must have the same length, the ordering of values
    in the lists is assumed align.

    The constructed grid will be regular, in the sense that the grid spacing is
    constant. However, the resolution in each direction can be different,
    allowing for finer resolution in some direction.

    Parameters
    ----------
    resolution : float | list[float]
        Resolution of the grid. Can be a single resolution value that will be
        applied to all coordinates, or a list of values mapping a resolution
        value to each of the coordinates.
    bounds : list[tuple[float, float]]
        A list of bounds of the form `(lower_bound, upper_bound)` indicating
        the bounding box of the returned grid. For a grid between -180 and 180
        degrees in Longitude, one would set the bounds in this dimension to
        `(-180, 180)` (if using "left" for the 'definition' argument).
    coord_names : list[str]
        List of coordinate names in the same order as the bounds and
        resolution(s).
    definition : str ("left" | "center")
        Each grid box is defined by the centre of the box in each coordinate.
        This argument defines whether the lower bound in each coordinate
        direction defines the "left"-edge or the "center" of the 1st grid-box.

        - "left": The lower-bound can be used to define the left-side of each
          grid-box. The grid-box value will be the left-edge + 1/2 of resolution
          in each direction.
        - "center": The lower-bound can be used to define the center of each
          grid-box. The grid-box value will be the center in each direction

    Returns
    -------
    grid : xarray.DataArray:
        The grid defined by the resolution and bounding box.

    Examples
    --------
    >>> grid_from_resolution(
            resolution=5,
            bounds=[(-87.5, 90), (-177.5, 180)],  # Lower bound is centre
            coord_names=["lat", "lon"]
        )
    <xarray.DataArray (lat: 36, lon: 72)> Size: 21kB
    array([[nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan],
           ...,
           [nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan]], shape=(36, 72))
    Coordinates:
      * lat      (lat) float64 288B -87.5 -82.5 -77.5 ... 77.5 82.5 87.5
      * lon      (lon) float64 576B -177.5 -172.5 ... 172.5 177.5
    """
    if not isinstance(resolution, Iterable):
        resolution = [resolution for _ in range(len(bounds))]
    if len(resolution) != len(coord_names) or len(bounds) != len(coord_names):
        raise ValueError("Input lists must have the same length")

    if definition.lower() == "left":
        # The lower-bound is the left-most edge of the grid, so we need to shift
        # by 1/2 of the resolution in each direction to determine the grid-box
        # centres.
        bounds = [
            (lbound + res / 2, rbound)
            for (lbound, rbound), res in zip(bounds, resolution)
        ]
    coords = {
        c_name: np.arange(lbound, ubound, res)
        for c_name, (lbound, ubound), res in zip(
            coord_names, bounds, resolution
        )
    }
    grid = xr.DataArray(coords=xr.Coordinates(coords))
    return grid


def assign_to_grid(
    values: np.ndarray,
    grid_idx: np.ndarray,
    grid: xr.DataArray,
    fill_value: Any = np.nan,
) -> xr.DataArray:
    """
    Assign a vector of values to a grid, using a list of grid index values.

    Parameters
    ----------
    values : numpy.ndarray
        The values to map onto the output grid.
    grid_idx : numpy.ndarray
        The 1d index of the grid (assuming "C" style ravelling) for each value.
    grid : xarray.DataArray
        The grid used to define the output grid.
    fill_value : Any
        The value to fill unassigned grid boxes. Must be a valid value of the
        input `values` data type.

    Returns
    -------
    out_grid : xarray.DataArray
        A new grid containing the values mapped onto the grid.
    """
    values = values.reshape(-1)
    grid_idx = grid_idx.reshape(-1)

    # Check that the fill_value is valid
    values_dtype = values.dtype
    fill_value_dtype = type(fill_value)
    if not np.can_cast(fill_value_dtype, values_dtype):
        raise TypeError(
            f"Type of input 'fill_value' ({fill_value}: {fill_value_dtype}) "
            + f"is not valid for values data type: {values_dtype}."
        )

    out_grid = xr.DataArray(
        data=np.full(grid.shape, fill_value=fill_value, dtype=values_dtype),
        coords=grid.coords,
    )
    coords_to_assign = np.unravel_index(grid_idx, out_grid.shape, "C")
    out_grid.values[coords_to_assign] = values

    return out_grid


def grid_to_distance_matrix(
    grid: xr.DataArray,
    dist_func: Callable = haversine_distance_from_frame,
    **dist_kwargs,
) -> xr.DataArray:
    """
    Calculate a distance matrix between all positions in a grid. Orientation of
    latitude and longitude will be maintained in the returned distance matrix.

    Parameters
    ----------
    grid : xarray.DataArray
        A 2-d grid containing latitude and longitude indexes specified in
        decimal degrees.
    dist_func : Callable
        Distance function to use to compute pairwise distances. See
        glomar_gridding.distances.calculate_distance_matrix for more
        information.
    **dist_kwargs
        Keyword arguments to pass to the distance function. This may include
        requirements for the name of specific coordinates, for example
        "lat_coord" and "lon_coord".

    Returns
    -------
    dist : xarray.DataArray
        A DataArray containing the distance matrix with coordinate system
        defined with grid cell index ("index_1" and "index_2"). The coordinates
        of the original grid are also kept as coordinates related to each
        index (the coordinate names are suffixed with "_1" or "_2"
        respectively).

    Examples
    --------
    >>> grid = grid_from_resolution(
            resolution=5,
            bounds=[(-90, 90), (-180, 180)],  # Lower bound is left-edge
            coord_names=["lat", "lon"],
            definition="left",
        )
    >>> grid_to_distance_matrix(grid, lat_coord="lat", lon_coord="lon")
    <xarray.DataArray 'dist' (index_1: 2592, index_2: 2592)> Size: 54MB
    array([[    0.        ,    24.24359308,    48.44112457, ...,
            19463.87158499, 19461.22915012, 19459.64166305],
           [   24.24359308,     0.        ,    24.24359308, ...,
            19467.56390938, 19463.87158499, 19461.22915012],
           [   48.44112457,    24.24359308,     0.        , ...,
            19472.29905588, 19467.56390938, 19463.87158499],
           ...,
           [19463.87158499, 19467.56390938, 19472.29905588, ...,
                0.        ,    24.24359308,    48.44112457],
           [19461.22915012, 19463.87158499, 19467.56390938, ...,
               24.24359308,     0.        ,    24.24359308],
           [19459.64166305, 19461.22915012, 19463.87158499, ...,
               48.44112457,    24.24359308,     0.        ]],
          shape=(2592, 2592))
    Coordinates:
      * index_1  (index_1) int64 21kB 0 1 2 3 4 ... 2587 2588 2589 2590 2591
      * index_2  (index_2) int64 21kB 0 1 2 3 4 ... 2587 2588 2589 2590 2591
        lat_1    (index_1) float64 21kB -87.5 -87.5 -87.5 ... 87.5 87.5
        lon_1    (index_1) float64 21kB -177.5 -172.5 ... 172.5 177.5
        lat_2    (index_2) float64 21kB -87.5 -87.5 -87.5 ... 87.5 87.5 87.5
        lon_2    (index_2) float64 21kB -177.5 -172.5 ... 172.5 177.5
    """
    coords = grid.coords
    out_coords = cross_coords(coords)

    dist: np.ndarray = calculate_distance_matrix(
        pl.DataFrame(
            {coord: out_coords[f"{coord}_1"].values for coord in coords}
        ),
        dist_func=dist_func,
        **dist_kwargs,
    )

    return xr.DataArray(
        dist,
        coords=xr.Coordinates(out_coords),
        name="dist",
    )


def cross_coords(
    coords: xr.Coordinates | xr.Dataset | xr.DataArray,
) -> xr.Coordinates:
    """
    Combine a set of coordinates into a cross-product, for example to construct
    a coordinate system for a distance matrix.

    For example a coordinate system defined by:
        lat = [0, 1],
        lon = [4, 5],
    would yield a new coordinate system defined by:
        index_1 = [0, 1, 2, 3]
        index_2 = [0, 1, 2, 3]
        lat_1 = [0, 0, 1, 1]
        lon_1 = [4, 5, 4, 5]
        lat_2 = [0, 0, 1, 1]
        lon_2 = [4, 5, 4, 5]

    Parameters
    ----------
    coords : xarray.Coordinates | xarray.DataArray | xarray.Dataset
        The set of coordinates to combine, or cross. This should be of length
        2 and have names defined by `lat_coord` and `lon_coord` input arguments.
        The ordering of the coordinates will define the cross ordering. If an
        array is provided then the coordinates are extracted.

    Returns
    -------
    cross_coords : xarray.Coordinates
        The new crossed coordinates, including index, and each of the input
        coordinates, for each dimension.

    Examples
    --------
    >>> grid = grid_from_resolution(
            resolution=5,
            bounds=[(-87.5, 90), (-177.5, 180)],  # Lower bound is centre
            coord_names=["lat", "lon"]
        )
    >>> cross_coords(grid.coords)
    Coordinates:
      * index_1  (index_1) int64 21kB 0 1 2 3 4 ... 2587 2588 2589 2590 2591
      * index_2  (index_2) int64 21kB 0 1 2 3 4 ... 2587 2588 2589 2590 2591
        lat_1    (index_1) float64 21kB -87.5 -87.5 -87.5 ... 87.5 87.5
        lon_1    (index_1) float64 21kB -177.5 -172.5 ... 172.5 177.5
        lat_2    (index_2) float64 21kB -87.5 -87.5 -87.5 ... 87.5 87.5 87.5
        lon_2    (index_2) float64 21kB -177.5 -172.5 ... 172.5 177.5
    """
    if isinstance(coords, (xr.DataArray, xr.Dataset)):
        coords = coords.coords
    dims = coords.dims

    coord_df = pl.from_records(
        list(coords.to_index()),
        schema=list(dims),  # type: ignore
        orient="row",
    )

    n = coord_df.height
    cross_coords: dict[str, Any] = {"index_1": range(n), "index_2": range(n)}
    for i in range(1, 3):
        cross_coords.update(
            {f"{c}_{i}": (f"index_{i}", coord_df[c]) for c in coord_df.columns}
        )

    return xr.Coordinates(cross_coords)


class Grid:
    """
    Grid Class.

    Allows for construction of a grid to use in the Kriging process. A mask
    can be applied to the grid with the `add_mask` method. If a mask is applied
    then computed fields will account for the mask. This has the ability to
    improve performance of the Kriging process as masked positions will not be
    infilled.

    The class is constructed from an instance of `xarray.Coordinates`, however
    it can also be constructed using the `from_resolution` method, which
    replicates the behaviour of
    :py:func:`glomar_gridding.grid.grid_from_resolution`.

    Parameters
    ----------
    coords : xarray.Coordinates
        The coordinates of the output grid.

    Examples
    --------
    >>> grid = Grid(coordinates)
    >>> grid.add_mask(mask)
    >>> grid.distance_matrix()  # Haversine by default
    >>> grid.covariance("matern", psill=0.36, range=1300, nugget=0, nu=1.5)
    >>> grid.map_observations(obs)
    >>> grid.kriging("ordinary", error_covariance)
    >>> masked_infilled = grid.krige.solve()
    >>> infilled = grid.assign(masked_infilled)
    """

    is_masked: bool = False

    def __init__(
        self,
        coords: xr.Coordinates,
    ) -> NoneType:
        self.grid = xr.DataArray(coords=coords)
        return None

    @classmethod
    def from_resolution(
        cls,
        resolution: float | list[float],
        bounds: list[tuple[float, float]],
        coord_names: list[str],
        definition: Literal["center", "left"] = "center",
    ):
        """
        Generate a grid from a resolution value, or a list of resolutions for
        given boundaries and coordinate names.

        Note that all list inputs must have the same length, the ordering of
        values in the lists is assumed align.

        The constructed grid will be regular, in the sense that the grid spacing
        is constant. However, the resolution in each direction can be different,
        allowing for finer resolution in some direction.

        Parameters
        ----------
        resolution : float | list[float]
            Resolution of the grid. Can be a single resolution value that will
            be applied to all coordinates, or a list of values mapping a
            resolution value to each of the coordinates.
        bounds : list[tuple[float, float]]
            A list of bounds of the form `(lower_bound, upper_bound)` indicating
            the bounding box of the returned grid. For a grid between -180 and
            180 degrees in Longitude, one would set the bounds in this dimension
            to `(-180, 180)` (if using "left" for the 'definition' argument).
        coord_names : list[str]
            List of coordinate names in the same order as the bounds and
            resolution(s).
        definition : str ("left" | "center")
            Each grid box is defined by the centre of the box in each
            coordinate. This argument defines whether the lower bound in each
            coordinate direction defines the "left"-edge or the "center" of the
            1st grid-box.

            - "left": The lower-bound can be used to define the left-side of
              each grid-box. The grid-box value will be the left-edge + 1/2 of
              resolution in each direction.
            - "center": The lower-bound can be used to define the center of each
              grid-box. The grid-box value will be the center in each direction

        Returns
        -------
        grid : Grid
            The grid defined by the resolution and bounding box.

        Examples
        --------
        >>> grid = Grid.from_resolution(
                resolution=5,
                bounds=[(-87.5, 90), (-177.5, 180)],  # Lower bound is center
                coord_names=["lat", "lon"],
                definition="center",
            )
        >>> grid.grid
        <xarray.DataArray (lat: 36, lon: 72)> Size: 21kB
        array([[nan, nan, nan, ..., nan, nan, nan],
               [nan, nan, nan, ..., nan, nan, nan],
               [nan, nan, nan, ..., nan, nan, nan],
               ...,
               [nan, nan, nan, ..., nan, nan, nan],
               [nan, nan, nan, ..., nan, nan, nan],
               [nan, nan, nan, ..., nan, nan, nan]], shape=(36, 72))
        Coordinates:
          * lat      (lat) float64 288B -87.5 -82.5 -77.5 ... 77.5 82.5 87.5
          * lon      (lon) float64 576B -177.5 -172.5 ... 172.5 177.5
        """
        if not isinstance(resolution, Iterable):
            resolution = [resolution for _ in range(len(bounds))]
        if len(resolution) != len(coord_names) or len(bounds) != len(
            coord_names
        ):
            raise ValueError("Input lists must have the same length")

        if definition.lower() == "left":
            # The lower-bound is the left-most edge of the grid, so we need to
            # shift by 1/2 of the resolution in each direction to determine the
            # grid-box centres.
            bounds = [
                (lbound + res / 2, rbound)
                for (lbound, rbound), res in zip(bounds, resolution)
            ]

        coords = {
            c_name: np.arange(lbound, ubound, res)
            for c_name, (lbound, ubound), res in zip(
                coord_names, bounds, resolution
            )
        }
        return cls(xr.Coordinates(coords=coords))

    @property
    def size(self) -> int:
        """Size of grid"""
        return self.grid.size

    @property
    def coords(self) -> xr.Coordinates:
        """Grid coordinates"""
        return self.grid.coords

    @property
    def coord_names(self) -> list[str]:
        """Names of the coordinates"""
        return list(self.coords.dims)  # type: ignore

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the grid"""
        return self.grid.shape

    @property
    def grid_idx(self) -> np.ndarray:
        """All grid indices"""
        return np.arange(self.size)

    @property
    def coord_df(self) -> pl.DataFrame:
        """Convert the coordinates to a DataFrame"""
        return pl.from_records(
            list(self.coords.to_index()),
            schema=list(self.coord_names),
            orient="row",
        )

    @property
    def index_map(self) -> pl.DataFrame:
        """Get the mapping between mask and grid indices"""
        df = pl.DataFrame({"grid_idx": self.grid_idx})
        if self.is_masked and hasattr(self, "mask"):
            df = df.remove(self.mask.flatten().mask)  # type: ignore
        return df.with_row_index(name="mask_idx")

    def select_bounds(
        self,
        bounds: list[tuple[float, float]],
    ) -> NoneType:
        """
        Filter the Grid by a set of bounds. Updates the `grid` attribute.

        Parameters
        ----------
        bounds : list[tuple[float, float]]
            A list of tuples containing the lower and upper bounds for each
            dimension.
        """
        self.grid = select_bounds(self.grid, bounds, self.coord_names)

    def map_observations(
        self,
        obs: pl.DataFrame,
        obs_col: str,
        obs_coords: list[str] = ["lat", "lon"],
        sort: bool = True,
        bounds: list[tuple[float, float]] | None = None,
        add_grid_pts: bool = True,
        grid_prefix: str = "grid_",
        apply_mask: bool = True,
    ) -> pl.DataFrame:
        """
        Align an observation dataframe to the Grid.

        Maps observations to the nearest grid-point, and sorts the data by the
        1d index of the DataArray in a row-major format.

        The grid defined by the latitude and longitude coordinates of the input
        DataArray is then used as the output grid of the Gridding process.

        Parameters
        ----------
        obs : polars.DataFrame
            The observational DataFrame containing positional data with
            latitude, longitude values within the `obs_latname` and
            `obs_lonname` columns respectively. Observations are mapped to the
            nearest grid-point in the grid.
        obs_coords : list[str]
            Names of the column containing positional values in the input
            observational DataFrame.
        sort : bool
            Sort the observational DataFrame by the grid index
        bounds : list[tuple[float, float]] | None
            Optionally filter the grid and DataFrame to fall within spatial
            bounds. This list must have the same size and ordering as
            `obs_coords` and `grid_coords` arguments.
        add_grid_pts : bool
            Add the grid positional information to the observational DataFrame.
        grid_prefix : str
            Prefix to use for the new grid columns in the observational
            DataFrame.
        apply_mask : bool
            If the Grid is masked convert the grid index values to the masked
            grid values. This will drop any observations whose grid position is
            masked.

        Returns
        -------
        obs : pandas.DataFrame
            Containing additional `grid_*`, and `grid_idx` values
            indicating the positions and grid index of the observation
            respectively. The DataFrame is also sorted (ascendingly) by the
            `grid_idx` columns for consistency with the gridding functions.

        Examples
        --------
        >>> obs = pl.read_csv("/path/to/obs.csv")
        >>> grid = Grid.grid_from_resolution(
                resolution=5,
                bounds=[(-87.5, 90), (-177.5, 180)],  # Lower bound is centre
                coord_names=["lat", "lon"]
            )
        >>> obs = grid.map_observations(obs)
        """
        if bounds is not None:
            self.select_bounds(bounds)
            obs = filter_bounds(obs, bounds, obs_coords)

        grid_size = self.shape

        grid_idx: list[list[int]] = []
        obs_to_grid_pos: list[np.ndarray] = []
        for grid_coord, obs_coord in zip(self.coord_names, obs_coords):
            grid_pos = self.coords[grid_coord].values
            _grid_idx, _obs_to_grid_pos = find_nearest(
                grid_pos, obs[obs_coord].to_numpy()
            )
            grid_idx.append(_grid_idx)
            obs_to_grid_pos.append(_obs_to_grid_pos)
            del _grid_idx, _obs_to_grid_pos

        flattened_idx = np.ravel_multi_index(
            grid_idx,
            grid_size,
            order="C",  # row-major
        )

        grid_idx_col = grid_prefix + "idx"
        obs = obs.with_columns(pl.Series(grid_idx_col, flattened_idx))
        if add_grid_pts:
            obs = obs.with_columns(
                [
                    pl.Series(grid_prefix + obs_coord, grid_pos)
                    for grid_pos, obs_coord in zip(obs_to_grid_pos, obs_coords)
                ]
            )

        if sort:
            obs = obs.sort(grid_idx_col, descending=False)

        if apply_mask and self.is_masked:
            # Map to the masked indices (this will drop observations at masked
            # positions)
            obs = (
                obs.join(
                    self.index_map,
                    left_on=grid_idx_col,
                    right_on="grid_idx",
                    how="inner",
                    coalesce=True,
                )
                .drop(grid_idx_col)
                .rename({"mask_idx": grid_idx_col})
            )

        self.obs = obs.get_column(obs_col).to_numpy()
        self.idx = obs.get_column(grid_idx_col).to_numpy()

        return obs

    def assign_values(
        self,
        values: np.ndarray,
        grid_idx: np.ndarray | None = None,
        fill_value: Any = np.nan,
        apply_mask: bool = True,
    ) -> xr.DataArray:
        """
        Assign a vector of values to a grid, using a list of grid index values.

        Parameters
        ----------
        values : numpy.ndarray
            The values to map onto the output grid.
        grid_idx : numpy.ndarray | None
            The 1d index of the grid (assuming "C" style ravelling) for each
            value. If unset then it is assumed that the values are complete for
            the (possibly masked) grid.
        fill_value : Any
            The value to fill unassigned grid boxes. Must be a valid value of
            the input `values` data type.
        apply_mask : bool
            The input grid_idx represents the index for the masked grid, apply
            the mapping to the final grid index.

        Returns
        -------
        out_grid : xarray.DataArray
            A new grid containing the values mapped onto the grid.
        """
        values = values.flatten()
        # grid_idx = grid_idx or np.arange(self.index_map.height)
        if grid_idx is None and len(values) != self.index_map.height:
            raise ValueError(
                "length of values must match the size of the (masked) grid if "
                + "the 'grid_idx' input is `None`."
            )
        grid_idx = grid_idx if grid_idx is not None else np.arange(len(values))
        grid_idx = grid_idx.flatten()

        if apply_mask and self.is_masked:
            # Un-map from mapped grid index to final grid index
            grid_idx = (
                self.index_map.get_column("grid_idx").to_numpy().flatten()
            )[grid_idx]

        if len(grid_idx) != len(values):
            raise ValueError("Mismatch of masked input indices to grid")

        # Check that the fill_value is valid
        values_dtype = values.dtype
        fill_value_dtype = type(fill_value)
        if not np.can_cast(fill_value_dtype, values_dtype):
            raise TypeError(
                "Type of input 'fill_value' "
                + f"({fill_value}: {fill_value_dtype}) "
                + f"is not valid for values data type: {values_dtype}."
            )

        out_grid = xr.DataArray(
            data=np.full(
                self.grid.shape,
                fill_value=fill_value,
                dtype=values_dtype,
            ),
            coords=self.coords,
        )
        coords_to_assign = np.unravel_index(grid_idx, out_grid.shape, "C")
        out_grid.values[coords_to_assign] = values

        return out_grid

    def distance_matrix(
        self,
        dist_func: Callable = haversine_distance_from_frame,
        **dist_kwargs,
    ) -> NoneType:
        """
        Calculate a distance matrix between all positions in a grid. Orientation
        of latitude and longitude will be maintained in the returned distance
        matrix.

        One distances between unmasked coordinates will be calculated if the
        grid is masked.

        This sets the `dist` attribute.

        Parameters
        ----------
        grid : xarray.DataArray
            A 2-d grid containing latitude and longitude indexes specified in
            decimal degrees.
        dist_func : Callable
            Distance function to use to compute pairwise distances. See
            glomar_gridding.distances.calculate_distance_matrix for more
            information. Defaults to Haversine distances.
        **dist_kwargs
            Keyword arguments to pass to the distance function. This may include
            requirements for the name of specific coordinates, for example
            "lat_coord" and "lon_coord".

        Examples
        --------
        >>> grid = Grid.from_resolution(
                resolution=5,
                bounds=[(-87.5, 90), (-177.5, 180)],  # Lower bound is centre
                coord_names=["lat", "lon"]
            )
        >>> grid.to_distance_matrix(grid, lat_coord="lat", lon_coord="lon")
        >>> grid.dist
        <xarray.DataArray 'dist' (index_1: 2592, index_2: 2592)> Size: 54MB
        array([[    0.        ,    24.24359308,    48.44112457, ...,
                19463.87158499, 19461.22915012, 19459.64166305],
               [   24.24359308,     0.        ,    24.24359308, ...,
                19467.56390938, 19463.87158499, 19461.22915012],
               [   48.44112457,    24.24359308,     0.        , ...,
                19472.29905588, 19467.56390938, 19463.87158499],
               ...,
               [19463.87158499, 19467.56390938, 19472.29905588, ...,
                    0.        ,    24.24359308,    48.44112457],
               [19461.22915012, 19463.87158499, 19467.56390938, ...,
                   24.24359308,     0.        ,    24.24359308],
               [19459.64166305, 19461.22915012, 19463.87158499, ...,
                   48.44112457,    24.24359308,     0.        ]],
              shape=(2592, 2592))
        Coordinates:
          * index_1  (index_1) int64 21kB 0 1 2 3 4 ... 2587 2588 2589 2590 2591
          * index_2  (index_2) int64 21kB 0 1 2 3 4 ... 2587 2588 2589 2590 2591
            lat_1    (index_1) float64 21kB -87.5 -87.5 -87.5 ... 87.5 87.5
            lon_1    (index_1) float64 21kB -177.5 -172.5 ... 172.5 177.5
            lat_2    (index_2) float64 21kB -87.5 -87.5 -87.5 ... 87.5 87.5 87.5
            lon_2    (index_2) float64 21kB -177.5 -172.5 ... 172.5 177.5
        """
        out_coords = self._cross_coords()

        dist: np.ndarray = calculate_distance_matrix(
            pl.DataFrame(
                {
                    coord: out_coords[f"{coord}_1"].values
                    for coord in self.coord_names
                }
            ),
            dist_func=dist_func,
            **dist_kwargs,
        )

        self.dist = xr.DataArray(
            dist,
            coords=out_coords,
            name="dist",
        )

    def set_distance_matrix(
        self,
        distance_matrix: np.ndarray | xr.DataArray,
    ) -> NoneType:
        """
        Set a distance matrix. This is automatically adjusted if the grid is
        masked.

        Sets the `dist` attribute.

        Parameters
        ----------
        distance_matrix : numpy.ndarray | xarray.DataArray
            The distance matrix for the full (unmasked) grid.
        """
        self.dist = self.prep_covariance(distance_matrix)

    def covariance_matrix(
        self,
        variogram: Literal["exponential", "gaussian", "matern", "spherical"],
        **kwargs,
    ) -> NoneType:
        """
        Compute a covariance matrix from the grid using the distance matrix and
        a varigoram model, this requires the pre-computation of the distance
        matrix attribute (`dist`). If the grid is masked then the resulting
        `covariance` attribute (set by this method) will be filtered to align
        with the mask.

        This sets the `covariance` and `variogram` attributes.

        Parameters
        ----------
        variogram : str
            Lower-case name of the variogram model to use. One of "exponential",
            "gaussian", "matern", or "spherical". The `variogram` attribute will
            be set to an instance of the equivalent class from
            `glomar_gridding.variogram`.
        **kwargs
            Keyword arguments for the variogram model.
        """
        if not hasattr(self, "dist"):
            raise AttributeError(
                "Distance matrix must be computed or set before covariance "
                + "matrix is computed"
            )
        match variogram:
            case "exponential":
                self.variogram = ExponentialVariogram(**kwargs)
            case "gaussian":
                self.variogram = GaussianVariogram(**kwargs)
            case "matern":
                self.variogram = MaternVariogram(**kwargs)
            case "spherical":
                self.variogram = SphericalVariogram(**kwargs)
            case _:
                raise ValueError(
                    "Unexpected 'variogram' input, expected one of "
                    + "'exponential', 'gaussian', 'matern', or 'spherical'."
                )

        # Distance matrix is an xarray.DataArray so covariance is too.
        self.covariance: xr.DataArray = variogram_to_covariance(  # type: ignore
            self.variogram.fit(self.dist),
            variance=self.variogram.psill,
        )

    def set_covariance(
        self,
        covariance_matrix: np.ndarray | xr.DataArray,
    ) -> NoneType:
        """
        Set a covariance matrix. This is automatically adjusted if the grid is
        masked.

        Sets the `covariance` attribute.

        Parameters
        ----------
        covariance_matrix : numpy.ndarray | xarray.DataArray
            The covariance matrix for the full (unmasked) grid.
        """
        self.cov = self.prep_covariance(covariance_matrix)

    def _cross_coords(self) -> xr.Coordinates:
        """
        Combine a set of coordinates into a cross-product, for example to
        construct a coordinate system for a distance matrix.

        For example a coordinate system defined by:
            lat = [0, 1],
            lon = [4, 5],
        would yield a new coordinate system defined by:
            index_1 = [0, 1, 2, 3]
            index_2 = [0, 1, 2, 3]
            lat_1 = [0, 0, 1, 1]
            lon_1 = [4, 5, 4, 5]
            lat_2 = [0, 0, 1, 1]
            lon_2 = [4, 5, 4, 5]

        Returns
        -------
        cross_coords : xarray.Coordinates
            The new crossed coordinates, including index, and each of the input
            coordinates, for each dimension.

        Examples
        --------
        >>> grid = Grid.from_resolution(
                resolution=5,
                bounds=[(-87.5, 90), (-177.5, 180)],  # Lower bound is centre
                coord_names=["lat", "lon"]
            )
        >>> grid._cross_coords()
        Coordinates:
          * index_1  (index_1) int64 21kB 0 1 2 3 4 ... 2587 2588 2589 2590 2591
          * index_2  (index_2) int64 21kB 0 1 2 3 4 ... 2587 2588 2589 2590 2591
            lat_1    (index_1) float64 21kB -87.5 -87.5 -87.5 ... 87.5 87.5
            lon_1    (index_1) float64 21kB -177.5 -172.5 ... 172.5 177.5
            lat_2    (index_2) float64 21kB -87.5 -87.5 -87.5 ... 87.5 87.5 87.5
            lon_2    (index_2) float64 21kB -177.5 -172.5 ... 172.5 177.5
        """
        coord_df = self.coord_df

        if self.is_masked and hasattr(self, "mask"):
            coord_df = coord_df.remove(self.mask.flatten().mask)  # type: ignore

        n = coord_df.height
        cross_coords: dict[str, Any] = {
            "index_1": range(n),
            "index_2": range(n),
        }
        for i in [1, 2]:
            cross_coords.update(
                {
                    f"{c}_{i}": (f"index_{i}", coord_df[c])
                    for c in coord_df.columns
                }
            )

        return xr.Coordinates(cross_coords)

    def add_mask(
        self,
        mask: np.ndarray | np.ma.MaskedArray,
    ) -> NoneType:
        """
        Add and apply a mask to the grid.

        Parameters
        ----------
        mask : numpy.ndarray | numpy.ma.MaskedArray
            The mask to apply, either an array of Booleans or a masked array.
        """
        if not mask.shape == self.shape:
            raise ValueError("Mask must have the same dimensions as the grid.")

        self.is_masked = True
        if isinstance(mask, np.ma.MaskedArray):
            self.mask = mask
        elif mask.dtype == np.bool:
            self.mask = np.ma.masked_where(mask, mask)
        else:
            raise ValueError("Mask must be a masked array or array of booleans")

        self.masked_grid_idx = self.index_map.get_column("grid_idx")

    def remove_mask(self) -> NoneType:
        """Remove the mask"""
        self.is_masked = False
        if hasattr(self, "mask"):
            delattr(self, "mask")

    def prep_covariance(
        self,
        covariance_matrix: np.ndarray | xr.DataArray,
    ) -> np.ndarray | xr.DataArray:
        """
        Apply the mask to a covariance matrix. This could be used to resize an
        error covariance matrix ahead of masked kriging, for example.

        Parameters
        ----------
        covariance_matrix : numpy.ndarray | xarray.DataArray
            The covariance matrix.

        Returns
        -------
        numpy.ndarray
            The covariance matrix subset to the mask
        """
        if covariance_matrix.shape[0] == self.index_map.height:
            print(
                "Size of input Covariance Matrix already matches "
                + "the (masked) grid"
            )
            return covariance_matrix

        if covariance_matrix.shape[0] != self.size:
            raise ValueError("Mismatch between covariance size and grid size")

        if not self.is_masked:
            return covariance_matrix

        mask_idx = self.index_map.get_column("grid_idx").to_numpy()

        return covariance_matrix[mask_idx, :][:, mask_idx]

    def kriging(
        self,
        kriging_method: Literal["simple", "ordinary", "stochastic"],
        error_cov: np.ndarray | xr.DataArray | None = None,
    ) -> NoneType:
        """
        Add a Kriging class object to the grid. Allowing for easy Kriging over
        a masked grid.

        This sets the `krige` attribute with a instance of a
        :py:func:`glomar_gridding.kriging.Kriging` class object, which will have
        all of the attributes and methods of that class.

        Parameters
        ----------
        kriging_method : str
            Lower-case name of the Kriging class to use. One of "simple",
            "ordinary", or "stochastic".
        error_cov : numpy.ndarray | xarray.DataArray | None
            Optional error covariance matrix. This will apply a smoothing
            effect to the observation points (as well as over the infilling).
        """
        # Handle Error Covariance shape
        if error_cov is not None and error_cov.shape[0] == self.size:
            error_cov = self.prep_covariance(error_cov)
        if error_cov is not None and error_cov.shape != self.covariance.shape:
            raise ValueError(
                "Mismatch between Covariance and Error Covariance Shapes"
            )
        if isinstance(error_cov, xr.DataArray):
            error_cov = error_cov.values

        # Take observations as dataframe?
        match kriging_method:
            case "simple":
                self.krige = SimpleKriging(
                    covariance=self.covariance.values,
                    idx=self.idx,
                    obs=self.obs,
                    error_cov=error_cov,
                )
            case "ordinary":
                self.krige = OrdinaryKriging(
                    covariance=self.covariance.values,
                    idx=self.idx,
                    obs=self.obs,
                    error_cov=error_cov,
                )
            case "stochastic":
                if error_cov is None:
                    raise ValueError(
                        "Error Covariance is required for StochasticKriging"
                    )
                self.krige = StochasticKriging(
                    covariance=self.covariance.values,
                    idx=self.idx,
                    obs=self.obs,
                    error_cov=error_cov,
                )
            case _:
                raise ValueError(
                    "Unexpected 'kriging_method', expected one of "
                    + "'simple', 'ordinary', 'stochastic'"
                )
