import os

__all__ = ["simulate_background_stars"]

RA: int = 0
DEC: int = 1
FLUX: int = 2
BG_RA: float = 220.002540961 + 0.1
BG_DEC: float = -60.8330381775
ALPHA_CEN_FLUX: float = 1145.4129625806625
BG_WIN: float = 2.0 / 60.0
BG_RAD: float = 2.0 / 60.0 * np.sqrt(2.0)

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
    

def simulate_background_stars() -> None:
    """
    Sample the Gaia database for typical background stars.

    The primary use of this function is to
    build a sample that can be used to look for biases.
    """

    ra_in_range: float = np.abs(bg_stars_ra) < bg_win
    dec_in_range: float = np.abs(bg_stars_dec) < bg_win
    in_range: float = ra_in_range & dec_in_range

    bg_stars_ra_crop: float = bg_stars_ra[in_range]
    bg_stars_dec_crop: float = bg_stars_dec[in_range]
    bg_stars_flux_crop: float = bg_stars_flux[in_range]
    bg_stars_rel_flux_crop: float = bg_stars_flux_crop / alpha_cen_flux

    with open("toliman/assets/background.csv", "w") as sheet:
        sheet.write("ra,dec,rel_flux\n")
        for row in np.arange(sample_len):
            sheet.write(f"{bg_stars_ra_crop[row]},")
            sheet.write(f"{bg_stars_dec_crop[row]},")
            sheet.write(f"{bg_stars_rel_flux_crop[row]}\n")
