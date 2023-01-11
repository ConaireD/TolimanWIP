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

jax.config.update("jax_enable_x64", True)

npix: int = 256
detector_npix: int = 128


def downsample(arr: float, m: int) -> float:
    n: int = arr.shape[0]
    out: int = n // m

    dim_one: float = arr.reshape((n * out, m)).sum(1).reshape(n, out).T
    dim_two: float = dim_one.reshape((out * out, m)).sum(1).reshape(out, out).T
    
    return dim_two / m / m


mask: float = downsample(np.load('../component_models/sidelobes.npy'), 4)

central_wavelength: float = (595. + 695.) / 2.
aperture_diameter: float = .13
secondary_mirror_diameter: float = .032
detector_pixel_size: float = .375
width_of_struts: float = .01
number_of_struts: int = 3

# Created the aberrations on the aperture. 
shape: int = 5
nolls: list = np.arange(2, shape + 2, dtype=int)
coeffs: list = 1e-08 * jax.random.normal(jax.random.PRNGKey(0), (shape,))

true_separation: float = dl.utils.arcseconds_to_radians(8.)
true_position: float = np.array([0., 0.], dtype=float)
true_flux: float = 1e6
true_contrast: float = 2.
true_position_angle: float = 0.

alpha_centauri: object = dl.BinarySource( # alpha centauri
    position = true_position,
    flux = true_flux,
    contrast = true_contrast,
    separation = true_separation,
    position_angle = true_position_angle,
    wavelengths = 1e-09 * np.linspace(595., 695., 10, endpoint=True)
)

wavefront_factory: object = dl.CreateWavefront(
    npix, 
    aperture_diameter,
    wavefront_type = "Angular"
)

toliman_pupil: object = dl.StaticAperture(
    dl.CompoundAperture(
    [
            dl.UniformSpider(
                number_of_struts, 
                width_of_struts
            ),
            dl.AnnularAperture(
                aperture_diameter / 2., 
                secondary_mirror_diameter / 2.
            )
        ]
    ),
    npixels = npix,
    pixel_scale = aperture_diameter / npix
)

toliman_aberrations: object = dl.StaticAberratedAperture(
    dl.AberratedAperture(
         nolls, 
         coeffs, 
         dl.CircularAperture(
             aperture_diameter / 2.
         )
    ),
    coefficients = coeffs,
    npixels = npix,
    pixel_scale = aperture_diameter / npix
)

toliman_mask: object = dl.AddOPD(mask)
normalise: object = dl.NormaliseWavefront()

toliman_body: object = dl.AngularMFT(
    detector_npix,
    dl.utils.arcseconds_to_radians(detector_pixel_size)
)

toliman: object = dl.Optics(
    layers = [
        wavefront_factory,
        toliman_pupil,
        toliman_aberrations,
        toliman_mask,
        normalise,
        toliman_body,
        normalise
    ]
)


def pixel_response(shape: float, threshold: float, seed: int = 1) -> float:
    key: object = jax.random.PRNGKey(seed)
    return 1. + threshold * jax.random.normal(key, shape)


# +
toliman_jitter: object = dl.ApplyJitter(2.)
toliman_saturation: object = dl.ApplySaturation(2500.)
    
toliman_pixel_response: object = dl.ApplyPixelResponse(
    pixel_response((detector_npix, detector_npix), .05)
)

toliman_detector: object = dl.Detector(
    [toliman_pixel_response, toliman_jitter, toliman_saturation]
)
# -

model: object = dl.Instrument(
    optics = toliman,
    sources = [alpha_centauri],
    detector = toliman_detector
)

comp_model: callable = jax.jit(model.model)

psf: float = comp_model()

_: None = plots.plot_im_with_cax_on_fig(psf, plt.figure(figsize=(6, 6)))


def photon_noise(psf: float, seed: int = 0) -> float:
    key = jax.random.PRNGKey(seed)
    return jax.random.poisson(key, psf)


def latent_detector_noise(shape: float, seed: int = 0) -> float:
    key: object = jax.random.PRNGKey(seed)
    return jax.random.normal(key, shape)


toliman_photon_noise: float = photon_noise(psf)
mean_latent_noise: float = 100. # Why this number?
toliman_latent_noise: float = mean_latent_noise * latent_detector_noise(psf.shape)
toliman_image: float = toliman_photon_noise + toliman_latent_noise

_: None = plots.plot_im_with_cax_on_fig(toliman_image, plt.figure(figsize=(6, 6)))


@eqx.filter_jit
def loss(separation: float, *args) -> float:
    separation_path: str = "BinarySource.separation"
    separation: float = separation.squeeze()
    model, data = args
    simulation: float = model.set(separation_path, separation).model()
    return ((simulation - data) ** 2).sum()


initial_separation: float = np.array([dl.utils.arcseconds_to_radians(8.5)])

from jax.scipy import optimize

optimize.minimize(loss, initial_separation, (model, toliman_image), method = "BFGS")

help(optimize.minimize)


