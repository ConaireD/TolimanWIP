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

jax.config.update("jax_enable_x64", True)

mask: float = np.load('../component_models/sidelobes.npy')
npix: int = 256
detector_npix: int = 100


def downsample_by_m(arr: float, m: int) -> float:
    size: tuple = arr.shape[0]
    n: int = int(size / m)
    rows: list = []
    for i in range(n):
        cols: list = []
        for j in range(n):
            cols.append(np.mean(arr[(i * m):((i + 1) * m), (j * m):((j + 1) * m)]))
        rows.append(np.stack(cols))
    image: float = np.stack(rows)
    return image



jax.ops.segment_sum()

mask_: float = downsample_by_m(mask, 4)

plt.imshow(mask_)

mask_.shape

_: None = plots.plot_im_with_cax_on_fig(mask, fig=plt.figure(figsize=(4, 4)))

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

alpha_centauri: object = dl.BinarySource( # alpha centauri
    position = [0., 0.],
    flux = 1e6,
    contrast = 2.,
    separation = dl.utils.arcseconds_to_radians(8.),
    position_angle = 0.,
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
#     detector = toliman_detector
)

comp_model: callable = jax.jit(model.model)

psf: float = comp_model()



def photon_noise(psf: float, seed: int = 0) -> float:
    key = jax.random.PRNGKey(seed)
    return jax.random.poisson(key, psf)


def latent_detector_noise(shape: float, seed: int = 0) -> float:
    key: object = jax.random.PRNGKey(seed)
    return jax.random.normal(key, shape)


toliman_photon_noise: float = photon_noise(psf)
mean_latent_noise: float = 100. # Why this number?
toliman_latent_noise: float = mean_latent_noise * latent_detector_noise(psf)
toliman_image: float = toliman_photon_noise + toliman_latent_noise

_: None = plots.plot_im_with_cax_on_fig(toliman_image, plt.figure(figsize=(6, 6)))


@eqx.filter_jit
@eqx.filter_value_and_grad
def loss(data: float, model: float):
    forward_model: float = model.model()
    return (forward_model - data) ** 2

key = jax.random.PRNGKey(0)
position_vec = 0.05*jax.random.normal(key, (num_images,2))
separation_vec = 8 + 0.1*jax.random.normal(key, (1, num_images))
angle_vec = np.pi/2 + 0.02*jax.random.normal(key, (1,num_images))

zern_coeff_mat = jax.random.normal(key, (len(basis), num_images)) #scale is applied later

plt.figure()
plt.scatter(position_vec[:,0], position_vec[:,1], alpha = 0.5)
plt.title('position')

plt.figure()
plt.hist(separation_vec[0,:], bins = 15)
plt.title('separation')

plt.figure()
plt.hist(angle_vec[0,:], bins = 15)
plt.title('angle')"
# %%time
start_learning_rate = 4.5e-2
max_iter = 200

opt = optax.rmsprop(learning_rate=start_learning_rate)
flux = 1e12
gtol = 5e-5

estimated_pos_x = np.zeros(num_images)
estimated_pos_y = np.zeros(num_images)
estimated_sep   = np.zeros(num_images)
estimated_ang   = np.zeros(num_images)

zernikes = 'ApplyBasisOPD.coefficients'

plt.ion()
for i in range(num_images):
    # Define true/target params
    true_image_params = np.array([position_vec[i,0], position_vec[i,1], separation_vec[0,i], angle_vec[0,i]])
    true_zernike_params = zern_coeff_mat[:,i]
    true_params = np.concatenate((true_image_params, true_zernike_params))
    
    osys = osys.set(zernikes, true_zernike_params)
    # Create target image
    target_image = apply_photon_noise(make_image(true_params, osys)*flux)
    
    # Default starting params
    params = 1.1*true_params
    opt = optax.inject_hyperparams(optax.adam)(learning_rate=start_learning_rate)
    opt_state = opt.init(params)
    #model_osys = osys.set(zernikes, coeffs_init*1.05)
    
    # Do gradient descent
    lr = start_learning_rate
    for j in tqdm(range(max_iter)):
        grads = jax.grad(compute_loss)(params, osys, target_image)
        
        overall_gtol = np.sum(np.abs(grads))
        
        if overall_gtol < gtol:
            clear_output(wait=True)
            print('Gtol satisfied')
            break
        opt_state.hyperparams['learning_rate'] = lr
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        #model_osys = optax.apply_updates(model_osys, updates)
        #print(params/true_params)
        if j == max_iter - 1:
            clear_output(wait=True)
            print('Maximum iterations hit')
        
    estimated_params = params
    
    estimated_pos_x = estimated_pos_x.at[i].set(estimated_params[0])
    estimated_pos_y = estimated_pos_y.at[i].set(estimated_params[1])
    estimated_sep   = estimated_sep.at[i].set(estimated_params[2])
    estimated_ang   = estimated_ang.at[i].set(estimated_params[3])
    
    # Make plots
    
    print('Iteration: {}/{}'.format(i+1,num_images))
    plt.figure(figsize = (10,8))
    plt.subplot(2,2,1)
    plt.plot(np.abs(estimated_pos_x - position_vec[:,0])[:i+1], 'x')
    plt.hlines(np.mean(np.abs(estimated_pos_x - position_vec[:,0])[:i+1]), 0, i+1, ls = '--', color = 'black')
    plt.title('absolute x pos error')
    plt.yscale('log')

    plt.subplot(2,2,2)
    plt.plot(np.abs(estimated_pos_y - position_vec[:,1])[:i+1], 'x')
    plt.hlines(np.mean(np.abs(estimated_pos_y - position_vec[:,1])[:i+1]), 0, i+1, ls = '--', color = 'black')
    plt.title('absolute y pos error')
    plt.yscale('log')

    plt.subplot(2,2,3)
    plt.plot(np.abs(estimated_sep - separation_vec)[0,:i+1], 'x')
    plt.hlines(np.mean(np.abs(estimated_sep - separation_vec)[0,:i+1]), 0, i+1, ls = '--', color = 'black')
    plt.title('absolute seperation error')
    plt.yscale('log')

    plt.subplot(2,2,4)
    plt.plot(np.abs(estimated_ang - angle_vec)[0,:i+1], 'x')
    plt.hlines(np.mean(np.abs(estimated_ang - angle_vec)[0,:i+1]), 0, i+1, ls = '--', color = 'black')
    plt.title('absolute angle error')
    plt.yscale('log')
    plt.show()"
print(np.std(np.abs(estimated_sep - separation_vec)))
