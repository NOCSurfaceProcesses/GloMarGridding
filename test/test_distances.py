"""Tests of the distances module"""

from itertools import product
from math import sqrt

import numpy as np
import polars as pl
import pytest
from scipy.spatial.distance import mahalanobis

from glomar_gridding.distances import (
    calculate_distance_matrix,
    euclidean_distance,
    haversine_distance_from_frame,
    mahal_dist_func,
    sigma_rot_func,
    tau_variogram_from_frame,
)


def test_euclidean():
    R = 6371.0
    df = pl.DataFrame({"lat": [-90, 90, 0], "lon": [0, 0, 23]})

    dist = euclidean_distance(df, radius=R)
    print(dist)

    assert dist[0, 0] == dist[1, 1] == dist[2, 2] == 0.0
    assert dist[0, 1] == pytest.approx(2 * R)
    assert dist[0, 2] == pytest.approx(sqrt(2) * R)


def test_haversine():
    R = 6371.0
    halifax = (44.6476, -63.5728)
    southampton = (50.9105, -1.4049)
    expected = 4557  # From Google

    df = pl.from_records(
        [halifax, southampton], orient="row", schema=["lat", "lon"]
    )

    dist = haversine_distance_from_frame(df, radius=R)

    assert dist[0, 0] == dist[1, 1] == 0.0
    assert dist[0, 1] == pytest.approx(expected, abs=1)  # Allow 1km out

    dist2 = calculate_distance_matrix(df, radius=R)

    assert np.allclose(dist, dist2)


def test_mahalanobis():
    lats = 90 - 180 * np.random.rand(10)
    lons = 180 - 360 * np.random.rand(10)

    lat = 90 - 180 * np.random.rand()
    lon = 180 - 360 * np.random.rand()

    # NOTE: Cannot use displacements (as scipy function doesn't handle shifting
    #       around -pi, pi)
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    lats = np.deg2rad(lats)
    lons = np.deg2rad(lons)

    Lx = 1200
    Ly = 860
    theta = 5 * np.pi / 12

    sigma = sigma_rot_func(Lx, Ly, theta)
    assert sigma.size == 4
    si = np.linalg.inv(sigma)

    dy = lats - lat
    dx = lons - lon

    dists = mahal_dist_func(dx, dy, Lx, Ly, theta)
    for y, x, d in zip(lats, lons, dists):
        expected = mahalanobis(np.array([x, y]), np.array([lon, lat]), si)

        assert np.allclose(expected, d)

    N = 100
    lon_shifts = pl.Series("lon_shift", 0.5 - np.random.rand(N))
    lat_shifts = pl.Series("lat_shift", 0.5 - np.random.rand(N))
    olons = pl.int_range(-10, 11, step=5, eager=True)
    olats = pl.int_range(-10, 11, step=5, eager=True)

    lons = olons.sample(N, with_replacement=True).alias("grid_lon")
    lats = olats.sample(N, with_replacement=True).alias("grid_lat")

    n = olats.len()
    Lx_ = 2100 * np.random.rand(n**2)
    Ly_ = 1800 * np.random.rand(n**2)
    Lx = Lx_
    Ly = Ly_
    Lx[Ly_ > Lx_] = Ly_[Ly_ > Lx_]
    Ly[Ly_ > Lx_] = Lx_[Ly_ > Lx_]
    theta = np.pi - 2 * np.pi * np.random.rand(n**2)

    lat_grid, lon_grid = np.asarray(list(product(olats, olons))).T

    df2 = pl.DataFrame(
        {
            "grid_lat": lat_grid,
            "grid_lon": lon_grid,
            "grid_lx": Lx,
            "grid_ly": Ly,
            "grid_theta": theta,
        }
    )

    df = pl.DataFrame([lons, lats, lon_shifts, lat_shifts])
    df = df.with_columns(
        [
            (pl.col("grid_lon") + pl.col("lon_shift")).alias("lon"),
            (pl.col("grid_lat") + pl.col("lat_shift")).alias("lat"),
        ]
    )
    df = df.join(df2, on=["grid_lat", "grid_lon"], how="left")

    tau = tau_variogram_from_frame(df)

    assert tau.shape == (N, N)
