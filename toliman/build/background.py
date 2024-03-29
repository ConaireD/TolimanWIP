"""md
## Overview
Due to the nature of the PSF background stars have a good chance of 
affecting the science signal. In particular we are interested in the 
sidelobes although we haven't decided how to deal with these yet.
This file porvides an API for generating a typical sample of background 
stars using the Gaia database. 

## API
??? note "`load_background_stars`"
    ::: toliman.build.background.load_background_stars

??? note "`window_background_stars`"
    ::: toliman.build.background.window_background_stars

??? note "`flux_relative_to_alpha_cen`"
    ::: toliman.build.background.flux_relative_to_alpha_cen

??? note "`save_background_stars`"
    ::: toliman.build.background.save_background_stars
"""

import os
import jax.numpy as np
import toliman.constants as const
import toliman.build.paths as paths

__author__ = "Jordan Dennis"
__all__ = [
    "load_background_stars",
    "window_background_stars",
    "flux_relative_to_alpha_cen",
    "save_background_stars",
]

RA: int = 0
DEC: int = 1
FLUX: int = 2
BG_RA: float = 220.102540961
BG_DEC: float = -60.8330381775
ALPHA_CEN_FLUX: float = 1145.4129625806625
BG_WIN: float = 2.0 / 60.0

CONICAL_QUERY = """
SELECT
    TOP 12000 
    ra, dec, phot_g_mean_flux AS flux
FROM
    gaiadr3.gaia_source
WHERE
    CONTAINS(POINT('', ra, dec), CIRCLE('', {}, {}, {})) = 1 AND
    phot_g_mean_flux IS NOT NULL
"""

def load_background_stars(ra: float, dec: float, rad: float) -> float:
    """
    Retrieve a sample of backgound stars from the Gaia database.

    Selects the top 12000 stars and contains only the entries that 
    contain a measured flux. A word of caution is don't work near
    to zero as some of the ra/dec may wrap around to 360, which 
    causes things to break when recentering. This is not checked 
    programmatically, but is a painful experience.

    Parameters
    ----------
    ra: float, deg 
        The right ascension of section of sky to survey.
    dec: float, deg 
        The declination of the section of sky to survey.
    rad: float, deg 
        The radius of the conical region to survey.

    Returns
    -------
    background: float 
        A sample of positions (ra, dec), and fluxes of a preselected 
        background region of sky. The convention is RA along 0, DEC 
        along 1 and FLUX along 2.

    Examples
    --------
    ```python 
    >>> load_background_stars(3.0, 3.0, 2.0)
    ```
    """
    if ra <= rad:
        warnings.warn("`ra <= rad`. Coordinate wrapping may occur.")

    from astroquery.gaia import Gaia

    bg_stars: object = Gaia.launch_job(CONICAL_QUERY.format(ra, dec, rad))

    return np.array([
            np.array(bg_stars.results["ra"]) - ra,
            np.array(bg_stars.results["dec"]) - dec,
            np.array(bg_stars.results["flux"]),
        ], dtype = float)

def window_background_stars(background: float, width: float) -> float:
    """
    Get a square array of background stars.

    Parameters
    ----------
    background: float, [deg, deg, W/m/m]
        Coordinates (ra, dec), and fluxes of a set of stars. The
        conventions is [RA, DEC, FLUX] along the leading axis.
    width: float, deg
        The width of the square region to cut.

    Returns
    -------
    background_in_square: float, [deg, deg, W/m/m]
        The coordinates (ra, dec) and fluxes of a set of background 
        stars within a square.

    Examples
    --------
    >>> bgstars: float = load_background_stars(0.0, 0.0, 2.0)
    >>> windowed_bgstars: float = window_bg_stars(bgstars, 1.0)
    """
    in_width: float = np.abs(background[(0, 1), :]) < width
    logical_and: callable = lambda x: np.logical_and(x[0], x[1])
    in_range: float = np.apply_along_axis(logical_and, 0, in_width)
    return background[:, in_range]

def flux_relative_to_alpha_cen(background: float) -> float:
    """
    Convert the flux into relative units.

    Parameters
    ----------
    background: float, [deg, deg, W/m/m]
        The coordinates (ra, dec) and fluxes of a sample of background 
        stars. The convention is [RA, DEC, FLUX] along the leading 
        (zeroth) axis.

    Returns 
    -------
    background: float, [deg, deg, W/m/m]
        The coordinates (ra, dec) and fluxes of the sample. The same
        indexing convention is maintained but the units of the flux 
        are altered.

    Examples
    --------
    >>> bgstars: float = load_background_stars(0.0, 0.0, 2.0)
    >>> win_stars: float = window_background_stars(bgstars, 1.0)
    >>> rel_stars: float = flux_relative_to_alpha_cen(win_stars)
    """
    return np.array([
            background[RA], 
            background[DEC],
            background[FLUX] / ALPHA_CEN_FLUX
        ], dtype = float)

def save_background_stars(background: float, root: str) -> None:
    """
    Save the background stars to a prespecified location.

    In order to simplify the interface with programs written using 
    `toliman` all resource files are stored in a specfic directory
    `TOLIMAN_HOME`.

    Parameters
    ----------
    background: float, [deg, deg, W/m/m]
        The coordinates (ra, dec) and fluxes of a sample of background 
        stars. The indexing convention is [RA, DEC, FLUX] along the 
        leading axis.

    Examples
    --------
    >>> bgstars: float = load_background_stars(0.0, 0.0, 2.0)
    >>> win_stars: float = window_background_stars(bgstars, 1.0)
    >>> rel_stars: float = flux_relative_to_alpha_cen(win_stars)
    >>> save_background_stars(rel_stars)
    """
    if background.shape[0] != 3:
        raise ValueError("Invalid background stars.")
    if not os.path.isdir(root):
        os.mkdir(root)
    with open(paths.concat([root, "background.csv"]), "w") as sheet:
        sheet.write("ra,dec,rel_flux\n")
        for row in np.arange(background[RA].size):
            sheet.write("{},".format(background[RA][row]))
            sheet.write("{},".format(background[DEC][row]))
            sheet.write("{}\n".format(background[FLUX][row]))

def install_background_stars(
        root: str,
        ra: float = BG_RA, 
        dec: float = BG_DEC,
        width: float = BG_WIN,
    ) -> None:
    """
    Sample the Gaia database for typical background stars.

    This is a convinience function for setting up a saved sample of
    background stars. You may be wondering why we bother saving a 
    sample instead of reloading it each time? We made this decision 
    because it takes a while to load from the external database and
    is unlikely to be changed once you have an appropriate sample.
    Thus saving the sample decreases the startup time of programs 
    written using `toliman` at the expense of some disk space.

    Parameters
    ----------
    ra: float, deg
        The right ascension of the section of sky to sample. 
    dec: float, deg
        The declination of the section of sky to sample.
    width: float, deg
        The width of the square sample to take. 

    Examples
    --------
    >>> import toliman.constants as const
    >>> import os
    >>> simulate_background_stars()
    >>> os.path.isfile(const.get_const_as_type("BACKGROUND_DIR", str))
    ::: True
    """
    bgstars: float = load_background_stars(ra, dec, width / np.sqrt(2))
    win_stars: float = window_background_stars(bgstars, width)
    rel_stars: float = flux_relative_to_alpha_cen(win_stars)
    save_background_stars(rel_stars, root)

def are_background_stars_installed(root: str) -> bool:
    """
    Check if the background stars are installed.

    Parameters
    ----------
    root: str
        The directory to search for an installation in.

    Returns
    -------
    installed: bool
        True if the background stars are installed else false.

    Examples
    --------
    ```python 
    >>> import os
    >>> os.mkdir("tmp")
    >>> are_background_stars_installed("tmp")
    ::: False
    >>> open("tmp/background.csv", "w").close()
    >>> are_background_stars_installed("tmp")
    ::: True
    ```
    """
    if not os.path.isdir(root):
        return False

    BG_DIR: str = const.get_const_as_type("BACKGROUND_DIR", str)
    return os.path.isfile(BG_DIR)
