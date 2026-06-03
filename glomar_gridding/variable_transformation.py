"""Normality transformation of variables"""

import numpy as np
from scipy import stats


def weibull_2_normality(
    x: np.ndarray,
    c: float,
    use_kp11: bool = False,
):
    r"""
    Box Cox transform to make Weibull-distributed variable near normal.
    This is usually used for wind speeds.
    The transformation lambda parameter depends on the shape parameter.

    Reference: https://doi.org/10.2307/2287172

    .. math::
       \hat{x} = \frac{x^{\lambda} - 1}{\lambda}

    .. math::
       \lambda \approx 0.2654 \times \text{shape}
       \text{, if } x \sim \text{Weibull}(\text{shape}, \text{scale})

    A variation to the above is proposed by Kulkarni and Powar 2011:
    https://doi.org/10.1155/2011/863274

    .. math::
       \hat{x} = x^{\lambda}

    .. math::
       \lambda \approx 0.2776 \times \text{shape}

    Parameters
    ----------
    x: numpy.ndarray
        Variable that needs standardisation
    c: float
        Weibull shape parameter
    use_kp11: bool
        Use the approximation described in Kulkarni and Powar 2011

    Returns
    -------
    x_hat: numpy.ndarray
        Approximate normal transformed x
    """
    if use_kp11:
        lam = 0.2776 * c
        x_hat = x**lam
        return x_hat
    lam = 0.2654 * c
    x_hat = stats.boxcox(x, lmbda=lam)
    return x_hat

