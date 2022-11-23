from astroquery.gaia import Gaia
import jax.numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt

mpl.rcParams["text.usetex"] = True

conical_query = """
SELECT
    ra, dec, phot_g_mean_mag AS mag
FROM
    gaiadr3.gaia_source
WHERE
    CONTAINS(POINT('', ra, dec), CIRCLE('', {}, {}, {})) = 1
"""

alpha_cen_ra = 219.902540961
alpha_cen_dec = -60.8330381775
window_width = 10. / 60.

stars = Gaia.launch_job(conical_query.format(
    alpha_cen_ra, alpha_cen_dec, window_width))

ra = np.array(stars.results["ra"])
dec = np.array(stars.results["dec"])
raw_mag = stars.results["mag"]
max_mag = np.nanmax(raw_mag)
mag = np.where(np.isnan(raw_mag), max_mag, raw_mag)
ra_dist = np.abs(ra - alpha_cen_ra)
dec_dist = np.abs(dec - alpha_cen_dec)
alpha = (mag - max_mag) / mag.ptp()

plt.scatter(ra, dec, alpha=alpha + 1.)
plt.show()

