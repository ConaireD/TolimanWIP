import os

def _simulate_background_stars() -> None:
    """
    Sample the Gaia database for typical background stars.

    The primary use of this function is to
    build a sample that can be used to look for biases.
    """
    from astroquery.gaia import Gaia

    conical_query = """
    SELECT
        TOP 12000 
        ra, dec, phot_g_mean_flux AS flux
    FROM
        gaiadr3.gaia_source
    WHERE
        CONTAINS(POINT('', ra, dec), CIRCLE('', {}, {}, {})) = 1 AND
        phot_g_mean_flux IS NOT NULL
    """

    bg_ra: float = 220.002540961 + 0.1
    bg_dec: float = -60.8330381775
    alpha_cen_flux: float = 1145.4129625806625
    bg_win: float = 2.0 / 60.0
    bg_rad: float = 2.0 / 60.0 * np.sqrt(2.0)

    bg_stars: object = Gaia.launch_job(conical_query.format(bg_ra, bg_dec, bg_rad))

    bg_stars_ra: float = np.array(bg_stars.results["ra"]) - bg_ra
    bg_stars_dec: float = np.array(bg_stars.results["dec"]) - bg_dec
    bg_stars_flux: float = np.array(bg_stars.results["flux"])

    ra_in_range: float = np.abs(bg_stars_ra) < bg_win
    dec_in_range: float = np.abs(bg_stars_dec) < bg_win
    in_range: float = ra_in_range & dec_in_range
    sample_len: float = in_range.sum()

    bg_stars_ra_crop: float = bg_stars_ra[in_range]
    bg_stars_dec_crop: float = bg_stars_dec[in_range]
    bg_stars_flux_crop: float = bg_stars_flux[in_range]
    bg_stars_rel_flux_crop: float = bg_stars_flux_crop / alpha_cen_flux

    print(sample_len)

    with open("toliman/assets/background.csv", "w") as sheet:
        sheet.write("ra,dec,rel_flux\n")
        for row in np.arange(sample_len):
            sheet.write(f"{bg_stars_ra_crop[row]},")
            sheet.write(f"{bg_stars_dec_crop[row]},")
            sheet.write(f"{bg_stars_rel_flux_crop[row]}\n")
