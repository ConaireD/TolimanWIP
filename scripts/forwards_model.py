import jax.numpy as np
import matplotlib.pyplot as plt
import jax
import equinox as eqx
import os
import optax
import dLux as dl
import jax

from jax import grad
from tqdm.notebook import tqdm
from IPython.display import clear_output

jax.default_backend()\n"
# NOTE: THIS NOTEBOOK IS RUNNING 32-bit
This **greatly** increases computation time as GPUs prefer 32-bit, however this means flux **must** be limited to a maximum of 1e12 photons"
#jax.config.update(\"jax_enable_x64\", True)
r2a = dl.utils.radians_to_arcseconds
a2r = dl.utils.arcseconds_to_radians"
mask = np.load('TolimanMask_sidelobes.npy')"
# Generate mask and basic modelling parmaters

central_wav = (595+695)/2
wavels = 1e-9 * np.linspace(595, 695, 10) # Wavelengths
aperture_diameter = 0.12
arcsec_per_pixel = 0.375
pixel_scale_out = dl.utils.arcseconds_to_radians(arcsec_per_pixel)
det_npix = 100# 2048
wf_npix = 1024


position = np.array([0.0,0.0])
flux = 1
separation = dl.utils.arcseconds_to_radians(8.0)
position_angle = np.pi/2
wavelengths = wavels\n"
# Zernike Paramaters
basis = dl.utils.zernike_basis(10, wf_npix, outside=0.)[3:] #tip/tilt/piston not here (already simulated)
coeffs = 2e-8*jax.random.normal(jax.random.PRNGKey(0), [len(basis)])

zernike_layer = dl.ApplyBasisOPD(basis, coeffs)
osys = dl.utils.toliman(mask.shape[0], det_npix, detector_pixel_size=r2a(pixel_scale_out), extra_layers=[dl.AddOPD(mask), zernike_layer])"
def make_image(params, osys):
    zernikes = 'ApplyBasisOPD.coefficients'
    position = [a2r(params[0]), a2r(params[1])]
    separation = a2r(params[2])
    position_angle = params[3]
    zernike_params = 2e-8*params[4:]
    osys = osys.set(zernikes, zernike_params)

    
    source = dl.BinarySource(position , flux, separation, position_angle, wavelengths = wavelengths)
    image = osys.model(source=source)
    image /= np.sum(image)
    return image

@eqx.filter_jit
def compute_loss(params, osys, input_image):
    zernikes = 'ApplyBasisOPD.coefficients'
    image_params = params[:4]
    zernike_params = 2e-8*params[4:]
    osys = osys.set(zernikes, zernike_params)
    
    fmodel_image = make_image(params, osys)
    noise = np.sqrt(input_image)  # does this actually do anything?
    residual = (input_image - fmodel_image)/noise
    
    chi2 = np.sum(residual**2)
    return chi2

def apply_photon_noise(image, seed = 0):
    key = jax.random.PRNGKey(seed)
    image_noisy = jax.random.poisson(key = key,lam = image)
    return image_noisy

def apply_photon_noise(image, seed = 0):
    key = jax.random.PRNGKey(seed)
    image_noisy = jax.random.poisson(key = key,lam = image)
    image_noisy /= 1e5  # needed so next line doesn't cause underflow (?) errors for high fluxes
    image_noisy /= np.sum(image_noisy)
    return image_noisy"
num_images = 100

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
%%time
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

# time for 32 8m53s for 100
# time for 64 31m39s for 30

# note following table 32 bit had 100 iters, 64 had 30"
| Offset | STD x32   | STD x64   | STDx32 with zerns |
|--------|-----------|-----------|-------------------|
| 1%     | 3.676e-06 | 2.424e-06 | 4.094e-06         |
| 5%     | 7.752e-06 | 5.765e-06 | 6.007e-06         |
| 10%    | 1.035e-05 |    n/a    | 7.397e-06         |
| 25%    | 0.0678    |    n/a    | 0.392             |"