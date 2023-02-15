"""md
## Overview
`toliman` could have very, very many parameters. In the vast majority of
use cases we will only care about a few of these so it didn't make sense 
to force the user to mark the remainder as static. Instead I made many of 
these potential parameters constant, for example, the radius of the pupil.
There are other constants here as well, such as the metallicity of the 
two stars in the binary and the default resolutions.

By making them environment variables they can still be changed by the user.
Because this code is evaluated when the package is imported this means that 
changes to the constants need to be made from within `python` using the 
`set_const` function. 

## API
??? note "`set_const`"
    ::: toliman.constants.set_const

??? note "`is_const_defined`"
    ::: toliman.constants.is_const_defined

??? note "`get_const_as_type`"
    ::: toliman.constants.get_const_as_type

## Constants
| Name                                  | Value                         | Type      | Units     |
|---------------------------------------|-------------------------------|-----------|-----------|
| `DEFAULT_PUPIL_NPIX`                  | 256                           | `int`     | no.       |
| `DEFAULT_DETECTOR_NPIX`               | 128                           | `int`     | no.       |
| `DEFAULT_NUMBER_OF_ZERNIKES`          | 5                             | `int`     | no.       |
| `DEFAULT_MASK_DIR`                    | $TOLIMAN_HOME/mask.npy        | `str`     | none      |
| `SPECTRUM_DIR`                        | $TOLIMAN_HOME/spectra.csv     | `str`     | none      |
| `BACKGROUND_DIR`                      | $TOLIMAN_HOME/background.csv  | `str`     | none      |
| `TOLIMAN_PRIMARY_APERTURE_DIAMETER`   | 0.13                          | `float`   | meters    |
| `TOLIMAN_SECONDARY_MIRROR_DIAMETER`   | 0.032                         | `float`   | meters    |
| `TOLIMAN_DETECTOR_PIXEL_SIZE`         | 0.375                         | `float`   | arcsec    |
| `TOLIMAN_WIDTH_OF_STRUTS`             | 0.01                          | `float`   | meters    |
| `TOLIMAN_NUMBER_OF_STRUTS`            | 3                             | `int`     | no.       |
| `DEFAULT_DETECTOR_JITTER`             | 2.0                           | `float`   | pixels    |
| `DEFAULT_DETECTOR_SATURATION`         | 2500                          | `int`     | photons   |
| `DEFAULT_DETECTOR_THRESHOLD`          | 0.05                          | `float`   | norm.     |
| `ALPHA_CENTAURI_SEPARATION`           | 8.0                           | `float`   | arcsec    |
| `ALPHA_CENTAURI_POSITION`             | [0.0, 0.0]                    | `list`    | arcsec    |
| `ALPHA_CENTAURI_MEAN_FLUX`            | 1e5                           | `float`   | photons   |
| `ALPHA_CENTAURI_CONTRAST`             | 2.0                           | `float`   | none      |
| `ALPHA_CENTAURI_POSITION_ANGLE`       | 0.0                           | `float`   | radians   |
| `ALPHA_CEN_A_SURFACE_TEMP`            | 5790.0                        | `float`   | kelvin    |
| `ALPHA_CEN_A_METALICITY`              | 0.2                           | `float`   | norm.     |
| `ALPHA_CEN_A_SURFACE_GRAV`            | 4.0                           | `float`   | accel.    |
| `ALPHA_CEN_B_SURFACE_TEMP`            | 5260.0                        | `float`   | kelvin    |
| `ALPHA_CEN_B_METALICITY`              | 0.23                          | `float`   | norm.     |
| `ALPHA_CEN_B_SURFACE_GRAV`            | 4.37                          | `float`   | accel.    |
| `FILTER_MIN_WAVELENGTH`               | 595e-09                       | `float`   | meters    |
| `FILTER_MAX_WAVELENGTH`               | 695e-09                       | `float`   | meters    |
| `FILTER_DEFAULT_RES`                  | 24                            | `int`     | no.       |
"""

import os
import dLux as dl
import jax.numpy as np
import warnings

__author__: str = "Jordan Dennis"
__all__: list = [
    "set_const",
    "is_const_defined",
    "get_const_as_type",
]

def set_const(const: str, value: object) -> None:
    """
    Set the value of an environment variable.

    Gives a warning if the constant already has a value. In 
    general it is not recommended that you change the values,
    but it may sometimes be necessary. 

    Parameters
    ----------
    const: str 
        The name of the constant. See the documentation of use
        `list_consts` to view a complete list.
    value: object
        The value to assign to the constant.
    """
    if is_const_defined(const):
        warnings.warn("`{}` was already defined.".format(const))
    os.environ[const] = str(value)

def is_const_defined(const: str) -> bool:
    """
    Check if an environment variable has been set.

    Parameters
    ----------
    const: str
        The name of the constant.

    Returns
    -------
    defed: bool
        True if the constant is defined, else False.
    """
    if os.environ.get(const):
        return True
    return False

def get_const_as_type(const: str, t: type) -> type:
    """
    Retrieve a constant.

    Parameters
    ----------
    const: str
        The name of the constant.
    t: type
        The type of the constant.

    Returns
    -------
    value: t
        The value of the constant.
    """
    constant: str = os.environ[const]

    if not constant.startswith("["):
        return t(constant)

    values: list = [
        float(value.strip()) 
        for value in constant.strip("[").strip("]").split(",")
    ]

    return values

if not is_const_defined("TOLIMAN_HOME"):
    warnings.warn("`TOLIMAN_HOME` is not defined. Using `.assets`.")
    set_const("TOLIMAN_HOME", ".assets")

set_const("DEFAULT_PUPIL_NPIX", 256)
set_const("DEFAULT_DETECTOR_NPIX", 128)
set_const("DEFAULT_NUMBER_OF_ZERNIKES", 5)
set_const("DEFAULT_MASK_DIR", "{}/mask.npy".format(os.environ["TOLIMAN_HOME"]))
set_const("SPECTRUM_DIR", "{}/spectra.csv".format(os.environ["TOLIMAN_HOME"]))
set_const("BACKGROUND_DIR", "{}/background.csv".format(os.environ["TOLIMAN_HOME"]))
set_const("TOLIMAN_PRIMARY_APERTURE_DIAMETER", 0.13)
set_const("TOLIMAN_SECONDARY_MIRROR_DIAMETER", 0.032)
set_const("TOLIMAN_DETECTOR_PIXEL_SIZE", dl.utils.arcseconds_to_radians(0.375))
set_const("TOLIMAN_WIDTH_OF_STRUTS", 0.01)
set_const("TOLIMAN_NUMBER_OF_STRUTS", 3)
set_const("DEFAULT_DETECTOR_JITTER", 2.0)
set_const("DEFAULT_DETECTOR_SATURATION", 2500)
set_const("DEFAULT_DETECTOR_THRESHOLD", 0.05)
set_const("ALPHA_CENTAURI_SEPARATION", dl.utils.arcseconds_to_radians(8.0))
set_const("ALPHA_CENTAURI_POSITION", [0.0, 0.0])
set_const("ALPHA_CENTAURI_MEAN_FLUX", 1e5)
set_const("ALPHA_CENTAURI_CONTRAST", 2.0)
set_const("ALPHA_CENTAURI_POSITION_ANGLE", 0.0)
set_const("ALPHA_CEN_A_SURFACE_TEMP", 5790.0)
set_const("ALPHA_CEN_A_METALICITY", 0.2)
set_const("ALPHA_CEN_A_SURFACE_GRAV", 4.0)
set_const("ALPHA_CEN_B_SURFACE_TEMP", 5260.0)
set_const("ALPHA_CEN_B_METALICITY", 0.23)
set_const("ALPHA_CEN_B_SURFACE_GRAV", 4.37)
set_const("FILTER_MIN_WAVELENGTH", 595e-09)
set_const("FILTER_MAX_WAVELENGTH", 695e-09)
set_const("FILTER_DEFAULT_RES", 24)
set_const("BACKGROUND_STAR_SPEC_RES", 2)
