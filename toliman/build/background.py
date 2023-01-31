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

def load_background_stars() -> float:
    from astroquery.gaia import Gaia

    bg_stars: object = Gaia.launch_job(CONICAL_QUERY.format(BG_RA, BG_DEC, BG_RAD))

    return np.array([
            np.array(bg_stars.results["ra"]) - BG_RA,
            np.array(bg_stars.results["dec"]) - BG_DEC,
            np.array(bg_stars.results["flux"]),
        ], dtype = float)

def window_background_stars(bg_stars_ra: float, bg_stars_dec: float)

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
