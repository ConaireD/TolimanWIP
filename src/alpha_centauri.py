import dLux as dl 
import jax.numpy as np
import jax

ALPHA_CENTAURI_SEPARATION: float = dl.utils.arcseconds_to_radians(8.)
ALPHA_CENTAURI_POSITION: float = np.array([0., 0.], dtype=float)
ALPHA_CENTAURI_MEAN_FLUX: float = 1.
ALPHA_CENTAURI_CONTRAST: float = 2.
ALPHA_CENTAURI_POSITION_ANGLE: float = 0.

alpha_centauri: object = dl.BinarySource(
    position = true_position,
    flux = true_flux,
    contrast = true_contrast,
    separation = true_separation,
    position_angle = true_position_angle,
    wavelengths = 1e-09 * np.linspace(595., 695., 10, endpoint=True)
)

