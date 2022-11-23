from astroquery.gaia import Gaia
from multiprocessing import Pool 
import jax.numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt
import itertools as it

mpl.rcParams["text.usetex"] = True

conical_query = """
SELECT
    phot_g_mean_mag AS mag
FROM
    gaiadr3.gaia_source
WHERE
    CONTAINS(POINT('', ra, dec), CIRCLE('', {}, {}, {})) = 1
"""

alpha_cen_ra = 219.902540961
alpha_cen_dec = -60.8330381775
window_width = 2. / 60.

figure, axes = plt.subplots(5, 5, sharex=True, sharey=True)

def collect(ra: int, dec: int) -> None:
    ra_centre = alpha_cen_ra + ra * window_width
    dec_centre = alpha_cen_dec + dec * window_width
    region = Gaia.launch_job(conical_query.format(
        ra_centre, dec_centre, window_width))
    return np.array(region.results["mag"])

with Pool(16) as pool:
    centres = it.product(range(-2, 3), range(-2, 3))
    mags = pool.starmap(plot_tile_hist, centres)

max_size = max([mag.size for mag in mags])

for ra in range(5):
    for dec in range(5):
        stars = mags[ra + 5 * dec]
        axes[ra][dec].hist(stars, color="black")
        axes[ra][dec].set_title(stars.size)
        axes[ra][dec].set_facecolor((stars.size / max_size, 0, 0))

for i in range(5):
    axes[i][0].set_ylabel("{:.2f}".format((i - 2) * 2))
    axes[4][i].set_xlabel("{:.2f}".format((i - 2) * 2))

figure.supxlabel("Displacement (arcmin)")
figure.supylabel("Displacement (arcmin)")
plt.show()

