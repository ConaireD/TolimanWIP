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
alpha_cen_mag = 18.03996 
bg_ra = alpha_cen_ra + 0.1
bg_dec = alpha_cen_dec
bg_win = 2. / 60. 
bg_rad = 2. / 60. * np.sqrt(2.)

bg_stars = Gaia.launch_job(conical_query.format(bg_ra, bg_dec, bg_rad))

bg_stars_ra = np.array(bg_stars.results["ra"])
bg_stars_dec = np.array(bg_stars.results["dec"])
bg_stars_mag = np.array(bg_stars.results["mag"])

ra_in_range = np.abs(bg_stars_ra - bg_ra) < bg_win
dec_in_range = np.abs(bg_stars_dec - bg_dec) < bg_win
in_range = ra_in_range & dec_in_range

bg_stars_ra_crop = bg_stars_ra[in_range]
bg_stars_dec_crop = bg_stars_dec[in_range]
bg_stars_mag_crop = bg_stars_mag[in_range]

bg_stars_mag_no_nan_crop = np.where(np.isnan(bg_stars_mag_crop), np.nanmax(bg_stars_mag_crop), bg_stars_mag_crop)
alpha_crop = (bg_stars_mag_no_nan_crop - bg_stars_mag_no_nan_crop.min()) / bg_stars_mag_no_nan_crop.ptp()

fig_1 = plt.figure(constrained_layout=True)
axes = fig_1.add_gridspec(top=0.75, right=0.75).subplots()
axes.set(aspect=1)
axes_hist_x = axes.inset_axes([0, 1.05, 1, 0.25])
axes_hist_y = axes.inset_axes([1.05, 0, 0.25, 1])
axes.scatter(bg_stars_ra_crop, bg_stars_dec_crop, alpha=alpha_crop)
axes_hist_x.hist(bg_stars_ra_crop)
axes_hist_y.hist(bg_stars_dec_crop, orientation="horizontal")
plt.show()

fig_2 = plt.figure()
axes = plt.axes()
fig_2.add_axes(axes)
axes.hist(bg_stars_mag_crop)
axes.axvline(alpha_cen_mag, color="red")
axes.set_xlabel("phot_g_mean_mag")
axes.set_ylabel("count")
axes.set_title("Histogram of Magnitudes in the G Band")
plt.show()
