from astroquery.gaia import Gaia
import jax.numpy as np

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
window_width = 2. / 60.

test = Gaia.launch_job(
    conical_query.format(alpha_cen_ra, alpha_cen_dec, window_width))
print(dir(test))
print(test.results)
print(type(test.results))
print(dir(test.results))
print(np.array(test.results["mag"]))
