import pytest  # noqa: F401
from pathlib import Path

import numpy as np
import polars as pl

from glomar_gridding.spline import ThinPlateSpline, SphericalThinPlateSpline


def test_tps_spline() -> None:
    np.random.seed(314159)
    # TEST: lambda = 0 does not smooth
    n_samps = 100

    X = 1 - 2 * np.random.rand(n_samps, 2)
    X[:, 0] *= np.pi / 2
    X[:, 1] *= np.pi
    y = 0.5 - (
        np.sin(2 * X[:, 0])
        + np.cos(2 * X[:, 1])
        + 0.1 * np.random.randn(n_samps)
    )

    tps = ThinPlateSpline(X, y)
    y_0 = tps.fit(lam=0.0).y_fit

    assert np.allclose(y_0, y)

    stps = SphericalThinPlateSpline(X, y)
    y_0 = stps.fit(lam=0.0).y_fit

    assert np.allclose(y_0, y)
    return None


def test_gcv() -> None:
    np.random.seed(314159)
    n_samps = 100

    X = 1 - 2 * np.random.rand(n_samps, 2)
    X[:, 0] *= np.pi / 2
    X[:, 1] *= np.pi
    y = 0.5 - (
        np.sin(2 * X[:, 0])
        + np.cos(2 * X[:, 1])
        + 0.1 * np.random.randn(n_samps)
    )

    tps = ThinPlateSpline(X, y)
    lam = tps.estimate_lambda_gcv()
    assert lam is not None
    assert lam > 0
    return None


def test_predict() -> None:
    np.random.seed(314159)
    n_samps = 100

    X = 1 - 2 * np.random.rand(n_samps, 2)
    X[:, 0] *= np.pi / 2
    X[:, 1] *= np.pi
    y = 0.5 - (
        np.sin(2 * X[:, 0])
        + np.cos(2 * X[:, 1])
        + 0.1 * np.random.randn(n_samps)
    )
    X_grid = np.mgrid[-0.5:0.5:25j, -1:1:50j]
    X_test = X_grid.reshape(2, -1).T
    X_test *= np.pi

    tps = ThinPlateSpline(X, y)
    tps.fit(lam=0, set_results=True)
    y_pred, _ = tps.predict(X_test)
    assert y_pred.shape == (X_test.shape[0],)

    return None


def test_var() -> None:
    np.random.seed(314159)
    n_samps = 50

    X = 1 - 2 * np.random.rand(n_samps, 2)
    X[:, 0] *= np.pi / 2
    X[:, 1] *= np.pi
    y = 0.5 - (
        np.sin(2 * X[:, 0])
        + np.cos(2 * X[:, 1])
        + 0.1 * np.random.randn(n_samps)
    )

    tps = ThinPlateSpline(X, y)
    result = tps.fit()
    var = result.var

    assert isinstance(var, float)
    assert np.all(var > 0)


@pytest.mark.parametrize(
    "model, thresh",
    [
        (ThinPlateSpline, 1e-3),
        (SphericalThinPlateSpline, 5e-2),
    ],
)
def test_se(model, thresh):
    # TEST: against a result using R. Original data generated using
    #
    # np.random.seed(101)
    # n = 150
    # X = np.random.uniform(-5, 5, (n, 2))
    # y = (
    #     0.5 * X[:, 0]
    #     + np.exp(-(X[:, 0] ** 2 + X[:, 1] ** 2) / 2)
    #     + np.random.normal(0, 0.1, n)
    # )
    #
    # With grid defined:
    # grid_pts = np.linspace(-5, 5, 30)
    # gx, gy = np.meshgrid(grid_pts, grid_pts)
    # g_coords = np.column_stack([gx.ravel(), gy.ravel()])
    #
    # This data was run through R fields Tps(X, y, scale.type = "unscaled")

    data_path = Path(__file__).parent / "data" / "spline"
    data = pl.read_csv(data_path / "shared_data.csv")
    grid = pl.read_csv(data_path / "grid_coords.csv")
    r_res = pl.read_csv(data_path / "r_results.csv")

    X = data.select("x1", "x2").to_numpy()
    y = data.get_column("y").to_numpy()

    tps = model(X, y)
    tps.fit()
    py_gcv = tps.result.gcv

    # Sum of weights should be 0
    assert np.sum(tps.result.weights) < 1e-9

    g_coords = grid.to_numpy()

    # Python prediction
    py_preds, py_se = tps.predict(g_coords, compute_se=True)
    assert py_se is not None

    # Load R Results
    r_preds = r_res.get_column("grid_preds").to_numpy()
    r_se = r_res.get_column("grid_se").to_numpy()
    r_gcv = r_res.get_column("gcv").first()

    # Metrics
    pred_mse = np.mean((py_preds - r_preds) ** 2)
    se_mse = np.mean((py_se - r_se) ** 2)
    gcv_diff = abs(r_gcv - py_gcv)

    assert se_mse < 5e-4
    assert pred_mse < 5e-4
    assert gcv_diff < 1e-4

    assert np.allclose(py_se, r_se, atol=thresh)


@pytest.mark.parametrize(
    "model, thresh",
    [
        (ThinPlateSpline, 1e-3),
        (SphericalThinPlateSpline, 5e-2),
    ],
)
def test_se_zero_at_test_pts(model, thresh):
    # TEST: SE is approximately 0 at training points if lambda = 0
    #       Not expecting exact match - propagation of numerical/floating point
    #       errors, but should be close.
    np.random.seed(314159)
    n_samps = 100

    X_grid = np.mgrid[-0.5:0.5:25j, -1:1:50j]
    X_test = X_grid.reshape(2, -1).T
    X_test *= np.pi

    n = X_test.shape[0]
    idx = np.random.choice(np.arange(n), size=n_samps, replace=False)
    X = X_test[idx, :]
    y = 0.5 - (
        np.sin(2 * X[:, 0])
        + np.cos(2 * X[:, 1])
        + 0.1 * np.random.randn(n_samps)
    )

    tps = model(X, y)
    tps.fit(lam=0)

    py_preds, py_se = tps.predict(X_test, compute_se=True)

    assert np.allclose(py_preds[idx], y, atol=thresh), (
        f"Diff of obs values {np.max(np.abs(y - py_preds[idx])) = }"
    )
    assert np.allclose(py_se[idx], np.zeros(n_samps), atol=1e-6), (
        f"Non zero SE at obs: {np.max(np.abs(py_se[idx])) = }"
    )
