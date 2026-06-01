"""Normality transformation and standardisation variables"""

import numpy as np


def standardise_sample(x: np.ndarray) -> np.ndarray:
    """
    Standardise 1D variable x
    (i.e. minus mean, divide by sigma)
    May need to be used with loop or np.vectorize or alike
    to apply to multidimensional dataset

    Parameters
    ----------
    x: numpy.ndarray
        Variable that needs standardisation

    Returns
    -------
    x_hat: numpy.ndarray
        Standardised x
    """
    x_bar = np.nanmean(x)
    x_sigma = np.nanstd(x)
    x_hat = (x - x_bar) / x_sigma
    return x_hat


def box_cox_transformation(
    x: np.ndarray,
    ll: float) -> np.ndarray:
    """
    Apply Box-Cox transformation to x

    Parameters
    ----------
    x: numpy.ndarray
        Variable to be transformed
    ll: float
        Box-Cox lambda parameter

    Returns
    -------
    x_hat: numpy.ndarray
        Transformed x
    """
    if ll == 0:
        return np.log(x)
    upstairs = (x ** ll) - 1
    downstairs = ll
    return upstairs / downstairs


def weibull_2_normality_via_boxcox(
    x: np.ndarray,
    c: float,
    use_kp11: bool = False,
):
    r"""
    Box Cox transform to make Weibull-distributed
    variable (usually windspeed for our purposes) to normality.

    Reference: https://doi.org/10.2307/2287172

    .. math:: \hat{x} = \frac{x^{\lambda} - 1}{\lambda}
    .. math:: \lambda \approx 0.2654 \times \text{shape}  \text{, if } x \sim \text{Weibull}(\text{shape}, \text{scale})

    A variation of that method is proposed by Kulkarni and Powar 2011:
    https://doi.org/10.1155/2011/863274

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
        ll = 0.2776 * c
        x_hat = x ** ll
        return x_hat
    ll = 0.2654 * c
    x_hat = box_cox_transformation(x, ll)
    return x_hat
