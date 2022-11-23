from astroquery.gaia import Gaia
from multiprocessing import Pool 
import jax.numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt
import itertools as it

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

def plot_tile_hist(ra: int, dec: int) -> None:
    ra_centre = alpha_cen_ra + ra * window_width
    dec_centre = alpha_cen_dec + dec * window_width
    region = Gaia.launch_job(conical_query.format(
        ra_centre, dec_centre, window_width))
    return np.array(region.results["mag"])

with Pool(16) as pool:
    centres = it.product(range(-2, 3), range(-2, 3))
    mags = pool.starmap(plot_tile_hist, centres)

for ra in range(-2, 3):
    for dec in range(-2, 3):
        
