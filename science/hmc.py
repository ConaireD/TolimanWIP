import toliman
import toliman.math as math
import toliman.constants as const
import jax
import jax.numpy as np
import jax.random as jr
import jax.lax as jl
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpyro as npy
import numpyro.distributions as dist
import dLux 
import equinox
import os
import chainconsumer as cc

mpl.rcParams["text.usetex"] = True
mpl.rcParams['axes.titlesize'] = 20

os.chdir("".join(os.getcwd().partition("toliman")[:2]))

model: object = dLux.Instrument(
    optics = toliman.TolimanOptics(operate_in_static_mode = True),
    detector = toliman.TolimanDetector(),
    sources = [toliman.AlphaCentauri(), toliman.Background(number_of_bg_stars = 5)]
)

psf: float = model.model()
data: float = math.simulate_data(psf, 0.001* np.linalg.norm(psf))
fdata: float = data.flatten()

size: float = 5.0
figure: object = plt.figure(figsize = (2 * size, 2 * size))
subfigures: object = figure.subfigures(2, 1)

data_and_psf_figure: object = subfigures[0]
psf_axes: object = data_and_psf_figure.add_axes([0.0, 0.1, 0.4, 0.8])
psf_caxes: object = data_and_psf_figure.add_axes([0.4, 0.1, 0.05, 0.8])
psf_ticks: object = psf_axes.axis("off")
psf_frame: None = psf_caxes.set_frame_on(False)
psf_cmap: object = psf_axes.imshow(psf ** 1/3, cmap = plt.cm.inferno)
psf_cbar: object = data_and_psf_figure.colorbar(psf_cmap, cax = psf_caxes)
psf_title: object = psf_axes.set_title("$\\textrm{Model}$")
    
data_axes: object = data_and_psf_figure.add_axes([0.55, 0.1, 0.4, 0.8])
data_caxes: object = data_and_psf_figure.add_axes([0.95, 0.1, 0.05, 0.8])
data_ticks: object = data_axes.axis("off")
data_frame: None = data_caxes.set_frame_on(False)
data_cmap: object = data_axes.imshow(data ** 1/3, cmap = plt.cm.inferno)
data_cbar: object = data_and_psf_figure.colorbar(data_cmap, cax = data_caxes)
data_title: object = data_axes.set_title("$\\textrm{Data}$")

residuals_figure: object = subfigures[1]
residuals_axes: object = residuals_figure.add_axes([0.25, 0.1, 0.4, 0.8])
residuals_caxes: object = residuals_figure.add_axes([0.65, 0.1, 0.05, 0.8])
residuals_ticks: object = residuals_axes.axis("off")
residuals_frame: object = residuals_caxes.set_frame_on(False)
residuals_cmap: object = residuals_axes.imshow((psf - data) ** 1/3, cmap = plt.cm.inferno)
residuals_cbar: object = residuals_figure.colorbar(residuals_cmap, cax = residuals_caxes)
residuals_title: object = residuals_axes.set_title("$\\textrm{Residuals}$")

true_pixel_scale: float = const.get_const_as_type("TOLIMAN_DETECTOR_PIXEL_SIZE", float)
true_separation: float = const.get_const_as_type("ALPHA_CENTAURI_SEPARATION", float)

def hmc_model(model: object) -> None:
    position_in_pixels: float = npy.sample(
        "position_in_pixels", 
        dist.Uniform(-5, 5), 
        sample_shape = (2,)
    ) 
    position: float = npy.deterministic(
        "position", 
        position_in_pixels * true_pixel_scale
    )

    logarithmic_separation: float = npy.sample(
        "logarithmic_separation", 
        dist.Uniform(-5, -4)
    )
    separation: float = npy.deterministic(
        "separation", 
        10 ** (logarithmic_separation)
    )
    
#     x_projection: float = npy.sample("x_projection", dist.Normal(0, 1))
#     y_projection: float = npy.sample("y_projection", dist.HalfNormal(1))
#     position_angle: float = npy.deterministic(
#         "position_angle", 
#         np.arctan2(y_projection, x_projection)
#     )

    logarithmic_flux: float = npy.sample("logarithmic_flux", dist.Uniform(4, 6))
    flux: float = npy.deterministic("flux", 10 ** logarithmic_flux)

    logarithmic_contrast: float = npy.sample("logarithmic_contrast", dist.Uniform(-4, 2))
    contrast: float = npy.deterministic("contrast", 10 ** logarithmic_contrast)

    paths: list = [
        "BinarySource.position",
        "BinarySource.separation",
#         "BinarySource.position_angle",
        "BinarySource.flux",
        "BinarySource.contrast",
    ]
    
    values: list = [
        position,
        separation,
#         position_angle,
        flux,
        contrast,
    ]
        
    with npy.plate("data", len(fdata)):
        poisson_model: float = dist.Poisson(
            model.update_and_model("model", paths, values, flatten=True)
        )
        return npy.sample("psf", poisson_model, obs=fdata)

sampler = npy.infer.MCMC(
    npy.infer.NUTS(hmc_model),    
    num_warmup=500,
    num_samples=500,
    num_chains=1,
    progress_bar=True,
)

sampler.run(jr.PRNGKey(0), model, init_params = model)

samples: float = sampler.get_samples().copy()

logarithmic_contrast_samples: float = samples.pop("logarithmic_contrast")
logarithmic_flux_samples: float = samples.pop("logarithmic_flux")
logarithmic_separation_samples: float = samples.pop("logarithmic_separation")
# x_projection_samples: float = samples.pop("x_projection")
# y_projection_samples: float = samples.pop("y_projection")
position_in_pixels_samples: float = samples.pop("position_in_pixels")

samples.update({"x": samples.get("position")[:, 0], "y": samples.get("position")[:, 1]})
position: float = samples.pop("position")

out: object = cc.ChainConsumer().add_chain(samples).plotter.plot()
