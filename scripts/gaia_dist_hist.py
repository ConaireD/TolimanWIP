from astroquery.gaia import Gaia
import jax.numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt
import matplotlib.patches as patch

mpl.rcParams["text.usetex"] = True

conical_query = """
SELECT
    TOP 12000 
    ra, dec, phot_g_mean_mag AS mag
FROM
    gaiadr3.gaia_source
WHERE
    CONTAINS(POINT('', ra, dec), CIRCLE('', {}, {}, {})) = 1
"""

alpha_cen_ra = 219.902540961
alpha_cen_dec = -60.8330381775
alpha_cen_cur = (alpha_cen_ra, alpha_cen_dec)
window_width = 6. / 60.
radius = 2. / 60.
alpha_cen_prop_mot_ra = -3608
alpha_cen_prop_mot_dec = 686
alpha_cen_delta_ra = - alpha_cen_prop_mot_ra * .05 / 3600.
alpha_cen_delta_dec = - alpha_cen_prop_mot_dec * .05 / 3600.
alpha_cen_orig_ra = alpha_cen_ra - alpha_cen_delta_ra
alpha_cen_orig_dec = alpha_cen_dec - alpha_cen_delta_dec
alpha_cen_orig = (alpha_cen_orig_ra, alpha_cen_orig_dec)
alpha_cen_delta = (alpha_cen_delta_ra, alpha_cen_delta_dec)

stars = Gaia.launch_job(
    conical_query.format(
        alpha_cen_ra, alpha_cen_dec, window_width))

ra = np.array(stars.results["ra"])
dec = np.array(stars.results["dec"])
raw_mag = stars.results["mag"]
max_mag = np.nanmax(raw_mag)
mag = np.where(np.isnan(raw_mag), max_mag, raw_mag)
ra_dist = np.abs(ra - alpha_cen_ra)
dec_dist = np.abs(dec - alpha_cen_dec)
alpha = (mag - max_mag) / mag.ptp()

axes = plt.axes()
axes.scatter(ra, dec, alpha=alpha + 1.)
axes.add_patch(patch.Circle(alpha_cen_cur, radius=0.01, color="red"))
axes.set_aspect(1 / axes.get_data_ratio())
axes.arrow(*alpha_cen_orig, *alpha_cen_delta, color="pink")
plt.show()
