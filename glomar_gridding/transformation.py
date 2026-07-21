"""Transformation of variables including to approximately normal"""

import numpy as np
from scipy import special


def weibull_to_normality(
    x: np.ndarray,
    c: float,
    use_kp11: bool = False,
):
    r"""
    Box Cox transform to make Weibull-distributed variable to near normal.
    This includes wind speeds, which is often modelled with Weibull.
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
        Variable that needs transformation
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
    x_hat = special.boxcox(x, lam)
    return x_hat


def inv_weibull_to_normality(
    x_hat: np.ndarray,
    c: float,
    use_kp11: bool = False,
):
    """
    The inverse of weibull_to_normality

    Parameters
    ----------
    x_hat: numpy.ndarray
        Variable that have been transformed
    c: float
        Weibull shape parameter
    use_kp11: bool
        Use the approximation described in Kulkarni and Powar 2011

    Returns
    -------
    x: numpy.ndarray
        Inverse-transformed x_hat
    """
    if use_kp11:
        lam_inv = 1 / (0.2776 * c)
        x = x_hat**lam_inv
        return x
    lam = 0.2654 * c
    x = special.inv_boxcox(x_hat, lam)
    return x


def dew_point_to_vapor_pressure(
    dew_point: float,
    dew_point_is_in_k: bool = True,
):
    """
    Convert dew point to vapor pressure.
    Same equation for temperature to saturation vapor pressure.

    There are multiple formulas for this, depending on variation
    of parameters within the Clausius-Clapeyron relation. AirSeaFluxCode
    uses a simplified version of Buck (1981) (Eq 8):
    https://doi.org/10.1175/1520-0450(1981)020<1527:NEFCVP>2.0.CO;2
    per Biri et al (2023) section 2.3.

    Often called the Magnus formula.

    MetPy (a Python package) uses WMO
    https://library.wmo.int/records/item/41650-guide-to-instruments-and-methods-of-observation
    Volume 1, equation 4.3 (that equation works on mixing ratio)

    Note:
    Current version of airseaflux code says -32.19 if T in K
    The correct value is actually -32.18...

    Returned value is in Pa. The equation in Buck (1981) gives hPa.

    Parameters
    ----------
    dew_point: float
        Dew point (or temperature) value

    dew_point_is_in_k: bool
        Set to True if `dew point` is in Kelvins
        Default is True
        MAKE SURE YOU CHECK THIS.
        Buck 1981 uses degree centigrade.

    Returns
    -------
    vapor_pressure: float
        (Saturation) vapor pressure from the given temperature
        Saturation vapor pressure if dew_point is an actual temperature.
        Partial vapor pressure if dew_point is dew point.
        UNITS ARE IN Pascals (NOT hPa as in Buck 1981)
    """
    if dew_point_is_in_k:
        dew_point = dew_point - 273.15
    vapor_pressure = 611.21 * np.exp(
        (17.502 * dew_point) / (240.97 + dew_point)
    )
    return vapor_pressure


temperature_to_saturation_vapor_pressure = dew_point_to_vapor_pressure


def vapor_pressure_to_dew_point(
    vapor_pressure: float,
    output_in_k: bool = True,
):
    """
    Inverse of `dew_point_to_vapor_pressure` of Buck (1981)
    equation. While dew_point_to_vapor_pressure exists in AirSeaFlux code,
    the inverse is missing. We are going to unbuck the Buck equation;
    there will be no more buck passing. The buck stops here.

    Parameters
    ----------
    vapor_pressure: float
        Saturation or partial vapor pressure to H2O(g)

    output_in_k: bool
        Set to True if one wants `dew point` in Kelvins
        Default is True
        MAKE SURE YOU CHECK THIS.

    Returns
    -------
    dew_point: float
        Dew point or air temperature.
        In Kelvins if output_in_kelvin is True
    """
    b = (np.log(vapor_pressure) - np.log(611.21)) / 17.502
    h = 240.97 * b
    t = 1 - b
    dew_point = h / t
    if output_in_k:
        dew_point += 273.15
    return dew_point


saturation_vapor_pressure_to_temperature = vapor_pressure_to_dew_point


def vapor_pressure_to_specific_humidity(
    vapor_pressure: float,
    air_pressure: float,
    use_mixing_ratio_approximation: bool = False,
):
    """
    Converts vapor pressure to specific humidity.
    This function already exists in AirSeaFlux code in `qsat.py`,
    but the inverse does not exist (see `specific_humidity_to_vapor_pressure`)

    air_pressure and vapor_pressure should be in Pascals (not hPa)
    Output units g/kg

    Parameters
    ----------
    vapor_pressure: float
        Partial vapor pressure to H2O(g)

    air_pressure: float
        Air pressure in PASCALS (not hPa)

    use_mixing_ratio_approximation: bool
        Outputs mixing ratio instead of specific humidity

    Returns
    -------
    specific_humidity: float
        Specific humidity or mixing ratio depending on
        `use_mixing_ratio_approximation`. Units g/kg (MAKE SURE
        YOU CHECK THIS)
    """
    gas_const_frac = 0.622
    if use_mixing_ratio_approximation:
        return 1000.0 * gas_const_frac * vapor_pressure / air_pressure
    return (
        1000.0
        * gas_const_frac
        * vapor_pressure
        / (air_pressure - (1 - gas_const_frac) * vapor_pressure)
    )


def specific_humidity_to_vapor_pressure(
    specific_humidity: float,
    air_pressure: float,
    use_mixing_ratio_approximation: bool = False,
):
    """
    Inverse of `vapor_pressure_to_specific_humidity`

    Parameters
    ----------
    specific_humidity: float
        Specific humidity or mixing ratio depending on
        `use_mixing_ratio_approximation`. Units is
        g/kg (MAKE SURE YOU CHECK THIS)

    air_pressure: float
        Air pressure in PASCALS (not hPa)

    use_mixing_ratio_approximation: bool
        specific_humidity is mixing ratio

    Returns
    -------
    vapor_pressure: float
        Vapor pressure of H2O(g)
    """
    gas_const_frac = 0.622
    a = specific_humidity / (1000 * gas_const_frac)
    e_if_mixing_ratio = a * air_pressure
    if use_mixing_ratio_approximation:
        return e_if_mixing_ratio
    return e_if_mixing_ratio / (1 + a * (1 - gas_const_frac))


def dew_point_to_specific_humidity(
    dew_point: float,
    air_pressure: float,
    dew_point_is_in_k: bool = True,
    use_mixing_ratio_approximation: bool = False,
):
    """
    Dew point to specific humidity

    Parameters
    ----------
    dew_point: float
        Dew point or air temperature,
        units depend on dew_point_is_in_k,
        kelvins or degree centigrade,
        (MAKE SURE YOU CHECK THIS)

    air_pressure: float
        Air pressure in PASCALS (not hPa)

    dew_point_is_in_k: bool
        Set to True if one wants `dew point` in Kelvins
        Default is True
        MAKE SURE YOU CHECK THIS.

    use_mixing_ratio_approximation: bool
        specific_humidity is mixing ratio

    Returns
    -------
    specific_humidity: float
        Specific humidity or mixing ratio depending on the value of
        use_mixing_ratio_approximation. Units = g / kg
    """
    return vapor_pressure_to_specific_humidity(
        dew_point_to_vapor_pressure(
            dew_point, dew_point_is_in_k=dew_point_is_in_k
        ),
        air_pressure,
        use_mixing_ratio_approximation=use_mixing_ratio_approximation,
    )


def specific_humidity_to_dew_point(
    specific_humidity: float,
    air_pressure: float,
    use_mixing_ratio_approximation: bool = False,
    output_in_k: bool = True,
):
    """
    Specific humidity to dew_point

    Parameters
    ----------
    specific_humidity: float
        Specific humidity or mixing ratio depending on
        `use_mixing_ratio_approximation`. Units is
        g/kg (MAKE SURE YOU CHECK THIS)

    air_pressure: float
        Air pressure in PASCALS (not hPa)

    use_mixing_ratio_approximation: bool
        specific_humidity is mixing ratio

    output_in_k: bool
        Set to True if one wants `dew point` in Kelvins
        Default is True
        MAKE SURE YOU CHECK THIS.

    Returns
    -------
    dew_point: float
        Dew point temperature, in K if output_in_k is True
        otherwise degree centigrade
    """
    return vapor_pressure_to_dew_point(
        specific_humidity_to_vapor_pressure(
            specific_humidity,
            air_pressure,
            use_mixing_ratio_approximation=use_mixing_ratio_approximation,
        ),
        output_in_k=output_in_k,
    )
