# +
import jax.numpy as np
import matplotlib.pyplot as plt
import jax
import equinox as eqx
import os
import optax
import dLux as dl
import jax
import tqdm.notebook as tqdm
import plots
import functools
from jax.scipy import optimize

jax.config.update("jax_enable_x64", True)
# -

_: None = plots.plot_im_with_cax_on_fig(psf, plt.figure(figsize=(6, 6)))


def photon_noise(psf: float, seed: int = 0) -> float:
    key = jax.random.PRNGKey(seed)
    return jax.random.poisson(key, psf)


def latent_detector_noise(scale: float, shape: float, seed: int = 0) -> float:
    key: object = jax.random.PRNGKey(seed)
    return scale * jax.random.normal(key, shape)


rescaling: float = 1e5

toliman_photon_noise: float = photon_noise(rescaling * psf) / rescaling
toliman_latent_noise: float = latent_detector_noise(1.0 / rescaling, psf.shape)
toliman_image: float = toliman_photon_noise + toliman_latent_noise

_: None = plots.plot_im_with_cax_on_fig(toliman_image, plt.figure(figsize=(6, 6)))


@eqx.filter_jit
def loss(values: float, *args) -> float:
    separation_path: str = "BinarySource.separation"
    flux_path: str = "BinarySource.flux"
    contrast_path: str = "BinrarySource.contrast"
    separation: float = values[0]
    flux: float = values[1]
    model, data = args
    simulation: float = model.set(separation_path, separation).model()
    return ((simulation - data) ** 2).sum()


separation_path: str = "BinarySource.separation"
flux_path: str = "BinarySource.flux"
contrast_path: str = "BinrarySource.contrast"
separation: float = values[0]
flux: float = values[1]
model, data = args
simulation: float = model.set(separation_path, separation).model()

initial_separation: float = np.array([dl.utils.arcseconds_to_radians(8.5)])

results: object = optimize.minimize(
    loss, initial_separation, (model, toliman_image), method="BFGS"
)

results.x

true_separation
