from astroquery.gaia import Gaia
import jax.numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt
import matplotlib.patches as patch

mpl.rcParams["text.usetex"] = True

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

bg_ra = 220.002540961 + 0.1
bg_dec = -60.8330381775
alpha_cen_flux = 1145.4129625806625
bg_win = 2. / 60. 
bg_rad = 2. / 60. * np.sqrt(2.)

bg_stars = Gaia.launch_job(conical_query.format(bg_ra, bg_dec, bg_rad))

bg_stars_ra = np.array(bg_stars.results["ra"])
bg_stars_dec = np.array(bg_stars.results["dec"])
bg_stars_flux = np.array(bg_stars.results["flux"])

ra_in_range = np.abs(bg_stars_ra - bg_ra) < bg_win
dec_in_range = np.abs(bg_stars_dec - bg_dec) < bg_win
in_range = ra_in_range & dec_in_range
sample_len = in_range.sum()

bg_stars_ra_crop = bg_stars_ra[in_range]
bg_stars_dec_crop = bg_stars_dec[in_range]
bg_stars_flux_crop = bg_stars_flux[in_range]
bg_stars_rel_flux_crop = bg_stars_flux_crop / alpha_cen_flux

with open("datasheets/bg_stars.csv", "w") as sheet:
    sheet.write("ra,dec,rel_flux\n")
    for row in np.arange(sample_len):
        sheet.write(f"{bg_stars_ra_crop},")
        sheet.write(f"{bg_stars_dec_crop},")
        sheet.write(f"{bg_stars_rel_flux_crop}\n")
    
