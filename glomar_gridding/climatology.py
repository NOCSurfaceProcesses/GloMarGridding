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

"""Functions for mapping climatologies and computing anomalies"""

import polars as pl
import xarray as xr

from glomar_gridding.io import load_dataset
from glomar_gridding.utils import find_nearest, select_bounds


def join_climatology_by_doy(
    obs_df: pl.DataFrame,
    climatology_365: xr.Dataset,
    lat_col: str = "lat",
    lon_col: str = "lon",
    date_col: str = "date",
    var_col: str = "sst",
    clim_lat: str = "latitude",
    clim_lon: str = "longitude",
    clim_doy: str = "doy",
    clim_var: str = "climatology",
    temp_from_kelvin: bool = True,
) -> pl.DataFrame:
    """
    Merge a climatology from an xarray.DataArray into a polars.DataFrame using
    the **day of year** value and position.

    This function accounts for leap years by taking the average of the
    climatology values for 28th Feb and 1st March for observations that were
    made on the 29th of Feb.

    The climatology is merged into the DataFrame and anomaly values are
    computed.

    Parameters
    ----------
    obs_df : polars.DataFrame
        Observational DataFrame.
    climatology_365 : xarray.DataArray
        DataArray containing daily climatology values (for 365 days).
    lat_col : str
        Name of the latitude column in the observational DataFrame.
    lon_col : str
        Name of the longitude column in the observational DataFrame.
    date_col : str
        Name of the datetime column in the observational DataFrame. Day of year
        values are computed from this value.
    var_col : str
        Name of the variable column in the observational DataFrame. The merged
        climatology names will have this name prefixed to "_climatology", the
        anomaly values will have this name prefixed to "_anomaly".
    clim_lat : str
        Name of the latitude coordinate in the climatology DataArray.
    clim_lon : str
        Name of the longitude coordinate in the climatology DataArray.
    clim_doy : str
        Name of the day of year coordinate in the climatology DataArray.
    clim_var : str
        Name of the climatology variable in the climatology DataArray.
    temp_from_kelvin : bool
        Optionally adjust the climatology from Kelvin to Celsius if the variable
        is a temperature.

    Returns
    -------
    obs_df : polars.DataFrame
        With the climatology merged and anomaly computed. The new columns are
        "_climatology" and "_anomaly" prefixed by the `var_col` value
        respectively.
    """
    # Names of the output columns
    clim_var_name = f"{var_col}_climatology"
    anom_var_name = f"{var_col}_anomaly"

    climatology = pl.from_pandas(
        climatology_365[clim_var].to_dataframe().reset_index(drop=False)
    )
    if temp_from_kelvin:
        climatology = climatology.with_columns(
            (pl.col(clim_var) - 273.15).name.keep()
        )
    print(climatology)
    climatology = climatology.select(
        [clim_lat, clim_lon, clim_doy, pl.col(clim_var).alias(clim_var_name)]
    )
    # Ensure doy is a Day of Year value
    if climatology.get_column(clim_doy).dtype.is_temporal():
        climatology = climatology.with_columns(
            pl.col(clim_doy).dt.ordinal_day().name.keep()
        )

    obs_lat = obs_df.get_column(lat_col).to_numpy()
    _, lat_vals = find_nearest(climatology_365.coords[clim_lat].values, obs_lat)

    obs_lon = obs_df.get_column(lon_col).to_numpy()
    _, lon_vals = find_nearest(climatology_365.coords[clim_lon].values, obs_lon)

    obs_df = obs_df.with_columns(
        pl.Series("clim_lat", lat_vals),
        pl.Series("clim_lon", lon_vals),
    )

    mask = (obs_df.get_column(date_col).dt.is_leap_year()) & (
        obs_df.get_column(date_col).dt.ordinal_day().eq(60)
    )

    non_leap_df = (
        obs_df.filter(~mask)
        .with_columns(
            pl.col(date_col).dt.month().alias("mo"),
            pl.col(date_col).dt.day().alias("dy"),
        )
        .with_columns(
            pl.datetime(pl.lit(2009), pl.col("mo"), pl.col("dy"))
            .dt.ordinal_day()
            .alias("doy")
        )
        .join(
            climatology,
            left_on=["clim_lat", "clim_lon", "doy"],
            right_on=[clim_lat, clim_lon, clim_doy],
            how="left",
            coalesce=True,
        )
        .drop(["clim_lat", "clim_lon", "doy", "mo", "dy"])
    )

    # Take average of 28th Feb and 1st March for 29th Feb
    leap_clim = (
        climatology.filter(pl.col(clim_doy).is_between(59, 60, closed="both"))
        .group_by([clim_lat, clim_lon])
        .agg(pl.col(clim_var_name).mean())
    )

    leap_df = (
        obs_df.filter(mask)
        .join(
            leap_clim,
            left_on=["clim_lat", "clim_lon"],
            right_on=[clim_lat, clim_lon],
            how="left",
            coalesce=True,
        )
        .drop(["clim_lat", "clim_lon"])
    )

    del climatology, leap_clim

    obs_df = pl.concat([non_leap_df, leap_df])
    obs_df = obs_df.with_columns(
        (pl.col(var_col) - pl.col(clim_var_name)).alias(anom_var_name)
    )

    return obs_df


def read_climatology(
    clim_path: str,
    min_lat: float = -90,
    max_lat: float = 90,
    min_lon: float = -180,
    max_lon: float = 180,
    lat_var: str = "lat",
    lon_var: str = "lon",
    **kwargs,
) -> xr.Dataset:
    """
    Load a climatology dataset from a netCDF file.

    Parameters
    ----------
    clim_path : str
        Path to the climatology file. Can contain format blocks to be replaced
        by the values passed to kwargs.
    min_lat : float
        Minimum latitude to load.
    max_lat : float
        Maximum latitude to load.
    min_lon : float
        Minimum longitude to load.
    max_lon : float
        Maximum longitude to load.
    lat_var : str
        Name of the latitude variable.
    lon_var : str
        Name of the longitude variable.
    **kwargs
        Replacement values for the climatology path.

    Returns
    -------
    clim_ds : xarray.Dataset
        Containing the climatology bounded by the min/max arguments provided.
    """
    clim_ds: xr.Dataset = load_dataset(clim_path, **kwargs)
    clim_ds = select_bounds(
        clim_ds,
        bounds=[(min_lat, max_lat), (min_lon, max_lon)],
        variables=[lat_var, lon_var],
    )
    return clim_ds
