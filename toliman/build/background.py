import os

__all__ = ["simulate_background_stars"]

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
    contain a measured flux.

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
    >>> load_background_stars(0.0, 0.0, 2.0)
    """
    from astroquery.gaia import Gaia

    bg_stars: object = Gaia.launch_job(CONICAL_QUERY.format(ra, dec, rad))

    return np.array([
            np.array(bg_stars.results["ra"]) - BG_RA,
            np.array(bg_stars.results["dec"]) - BG_DEC,
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
    in_width: float = np.abs(background[(0, 1)]) < width
    in_range: float = np.apply_along_axis(np.logical_and, 0, in_width)
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

def save_background_stars(background: float) -> None:
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
    BG_DIR: str = const.get_const_as_type("BACKGROUND_DIR", str)

    with open(BG_DIR, "w") as sheet:
        sheet.write("ra,dec,rel_flux\n")
        for row in np.arange(sample_len):
            sheet.write("{},".format(background[RA]))
            sheet.write("{},".format(background[DEC]))
            sheet.write("{}\n".format(background[FLUX]))


def simulate_background_stars(/,
        ra: float = BG_RA, 
        dec: float = BG_DEC,
        width: float = BG_WIN
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
    save_background_stars(rel_stars)

