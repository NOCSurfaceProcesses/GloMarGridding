"""Compare GloMarGridding Kriging against PyKrige and GSTools.

Required packages:
pip install pykrige gstools

The same spatial covariance matrices are used for GloMarGridding
and GSTools. For PyKrige, the exponential variogram range is set to 3
* len_scale, to match PyKrige's parameterisation (gamma(h) = sill *
(1 - exp(-3 h / range))).

Tests -------------------

* Kriged estimates must agree between GMO, GStools and PyKrige

* With no observation error, the Kriging variance must agree exactly

* With uncorrelated observation error supplied, GSTools returns the same
error variance

"""

import warnings

import numpy as np
import pytest

from glomar_gridding.kriging import OrdinaryKriging

gs = pytest.importorskip("gstools")

rtol = 1e-7
atol = 1e-9


def _grid(nx: int = 12, ny: int = 9) -> tuple[np.ndarray, np.ndarray]:
    gx, gy = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
    return gx.ravel(), gy.ravel()


def _distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])


MODELS = {
    "exponential": lambda: gs.Exponential(dim=2, var=1.7, len_scale=2.5),
    "matern15": lambda: gs.Matern(dim=2, var=0.9, len_scale=3.0, nu=1.5),
    "spherical": lambda: gs.Spherical(dim=2, var=1.2, len_scale=6.0),
}


def _setup(model_name: str, seed: int, n_obs: int = 10):
    rng = np.random.default_rng(seed)
    x, y = _grid()
    model = MODELS[model_name]()
    cov = model.covariance(_distances(x, y))
    idx = np.sort(rng.choice(x.size, n_obs, replace=False))
    obs = rng.normal(loc=2.0, size=n_obs)
    return rng, x, y, model, cov, idx, obs


def _glomar_ok(cov, idx, obs, error_cov=None):
    error = None if error_cov is None else error_cov.copy()
    ok = OrdinaryKriging(cov, idx, obs, error_cov=error)
    ok.get_kriging_weights()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        var = ok.get_uncertainty() ** 2
    return ok.solve(), var


def _gstools_ok(model, x, y, idx, obs, **kwargs):
    krig = gs.krige.Ordinary(
        model, cond_pos=[x[idx], y[idx]], cond_val=obs, **kwargs
    )
    est, var = krig([x, y], mesh_type="unstructured", return_var=True)
    return np.asarray(est), np.asarray(var)


# GSTools
# --------------------------------------------------------------------------
@pytest.mark.parametrize("model_name", list(MODELS))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ok_vs_gstools_noerr(model_name, seed):
    """Test GMO Kriging output (value and uncertainty) against GStools"""
    _, x, y, model, cov, idx, obs = _setup(model_name, seed)
    est, var = _glomar_ok(cov, idx, obs)
    ref_est, ref_var = _gstools_ok(model, x, y, idx, obs, exact=True)
    np.testing.assert_allclose(est, ref_est, rtol=rtol, atol=atol)
    np.testing.assert_allclose(var, ref_var, rtol=rtol, atol=atol)


@pytest.mark.parametrize("model_name", list(MODELS))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ok_vs_gstools_err(model_name, seed):
    """Test GMO Kriging output (value and uncertainty) against GStools with an uncorrelated observation uncertainty supplied"""
    rng, x, y, model, cov, idx, obs = _setup(model_name, seed)
    err_var = rng.uniform(0.05, 0.3, idx.size)
    est, var = _glomar_ok(cov, idx, obs, error_cov=np.diag(err_var))
    ref_est, ref_var = _gstools_ok(
        model, x, y, idx, obs, exact=False, cond_err=err_var
    )
    np.testing.assert_allclose(est, ref_est, rtol=rtol, atol=atol)
    np.testing.assert_allclose(var, ref_var, rtol=rtol, atol=atol)


# PyKrige
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ok_vs_pykrige_noerr(seed):
    """Test GMO Kriging output (value and uncertainty) against PyKrige"""
    pykrige_ok = pytest.importorskip("pykrige.ok")
    _, x, y, model, cov, idx, obs = _setup("exponential", seed)
    est, var = _glomar_ok(cov, idx, obs)
    pk = pykrige_ok.OrdinaryKriging(
        x[idx],
        y[idx],
        obs,
        variogram_model="exponential",
        variogram_parameters={
            "sill": model.var,
            "range": 3.0 * model.len_scale,
            "nugget": 0.0,
        },
        exact_values=True,
    )
    ref_est, ref_var = pk.execute("points", x, y)
    np.testing.assert_allclose(est, np.asarray(ref_est), rtol=rtol, atol=atol)
    np.testing.assert_allclose(var, np.asarray(ref_var), rtol=rtol, atol=atol)
