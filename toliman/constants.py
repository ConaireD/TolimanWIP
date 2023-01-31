import os
import dLux as dl
import jax.numpy as np
import warnings

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
    return t(os.environ[const])

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
set_const("ALPHA_CENTAURI_MEAN_FLUX", 1.0)
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
