import jax.numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import dLux as dl
from numpy import random
import scipy
from jax import grad
import jax
import equinox as eqx
import os

import optax
from IPython.display import clear_output\n
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=5' # Use 8 CPU devices # might be causing a memory leak
jax.config.update('jax_enable_x64', True)
r2a = dl.utils.radians_to_arcseconds
a2r = dl.utils.arcseconds_to_radians
mask = np.load('TolimanMask_sidelobes.npy')
plt.imshow(mask)
plt.colorbar()
central_wav = (595+695)/2

# Generate mask and basic modelling parmaters
wavels = 1e-9 * np.linspace(595, 695, 5) # Wavelengths
aperture_diameter = 0.12
arcsec_per_pixel = 0.375
pixel_scale_out = dl.utils.arcseconds_to_radians(arcsec_per_pixel)
det_npix = 2048
wf_npix = 1024
osys = dl.utils.toliman(mask.shape[0], 100, detector_pixel_size=r2a(pixel_scale_out), extra_layers=[dl.AddOPD(mask)])
#source = dl.PointSource(wavelengths=wavels)
#osys = osys.set('AddOPD.opd', mask)
position = np.array([0.0,0.0])
flux = 1
separation = dl.utils.arcseconds_to_radians(8.0)
position_angle = np.pi/2
wavelengths = wavels

mysource = dl.BinarySource(position , flux, separation, position_angle, wavelengths = wavelengths)
%%time
psf = osys.model(source=mysource)
plt.imshow(psf)
print(np.sum(psf))
def make_image(params, osys):
    position = [a2r(params[0]), a2r(params[1])]
    separation = a2r(params[2])
    position_angle = params[3]
    
    source = dl.BinarySource(position , flux, separation, position_angle, wavelengths = wavelengths)
    image = osys.model(source=source)
    image /= np.sum(image)
    return image

@eqx.filter_jit
def compute_loss(params, osys, input_image):
    fmodel_image = make_image(params, osys)
    noise = np.sqrt(input_image)  # does this actually do anything?
    residual = (input_image - fmodel_image)/noise
    
    chi2 = np.sum(residual**2)
    return chi2

def apply_photon_noise(image, seed = 0):
    key = jax.random.PRNGKey(seed)
    image_noisy = jax.random.poisson(key = key,lam = image)
    return image_noisy\n
%%time
target_image = apply_photon_noise(make_image(np.array([0.1,0.13,8.003,(np.pi/2)*1.01]), osys)*1e12)
#target_image = make_image(np.array([0,15,8,np.pi/3]), osys)
target_image /= np.sum(target_image)
plt.imshow(target_image)
print(np.sum(target_image))
%%time
start_learning_rate = 1e-3
optimizer = optax.adam(start_learning_rate)

params = np.array([0,0,8,np.pi/2])
opt_state = optimizer.init(params)

true_params = np.array([0.1,0.13,8.003,(np.pi/2)*1.01])

for i in range(10):
    print(params)
    grads = jax.grad(compute_loss)(params, osys, target_image)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
print(params)
    
final_image = make_image(params, osys)
residual = target_image - final_image
mse = np.mean(residual**2)
print('{:.3e}'.format(mse))
print('{:.3e}'.format(np.abs(true_params[2] - params[2])))
%%time
start_learning_rate = 1e-2
optimizer = optax.rmsprop(start_learning_rate)
max_iter = 25

params = np.array([0,0,8,np.pi/2])
opt_state = optimizer.init(params)

true_params = np.array([0.1,0.13,8.003,(np.pi/2)*1.01])

for i in range(max_iter):
    if i == max_iter - 1:
        print('Maximum iterations hit')
    #print(params)
    grads = jax.grad(compute_loss)(params, osys, target_image)
    print(grads)
    print('{:.3e}'.format(np.abs(true_params[2] - params[2])))
    if np.max(np.abs(grads)) < 5e-2:
        print('Gtol satisfied')
        break
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
#print(params)
    
final_image = make_image(params, osys)
residual = target_image - final_image
mse = np.mean(residual**2)
print('{:.3e}'.format(mse))
print('{:.3e}'.format(np.abs(true_params[2] - params[2])))
print(np.abs(true_params - params))
num_images = 50

key = jax.random.PRNGKey(0)
position_vec = 0.05*jax.random.normal(key, (num_images,2))
separation_vec = 8 + 0.1*jax.random.normal(key, (1, num_images))
angle_vec = np.pi/2 + 0.02*jax.random.normal(key, (1,num_images))

plt.figure()
plt.scatter(position_vec[:,0], position_vec[:,1], alpha = 0.5)
plt.title('position')
 
plt.figure()
plt.hist(separation_vec[0,:], bins = 15)
plt.title('separation')

plt.figure()
plt.hist(angle_vec[0,:], bins = 15)
plt.title('angle')
start_learning_rate = 4.5e-2
max_iter = 50

# exponential_decay_scheduler = optax.exponential_decay(init_value=start_learning_rate, transition_steps=max_iter,
#                                                       decay_rate=0.9, transition_begin=1,
#                                                       staircase=False)

opt = optax.rmsprop(learning_rate=start_learning_rate)
flux = 1e12
gtol = 9e-4

estimated_pos_x = np.zeros(num_images)
estimated_pos_y = np.zeros(num_images)
estimated_sep   = np.zeros(num_images)
estimated_ang   = np.zeros(num_images)

plt.ion()
for i in range(num_images):
    # Define true/target params
    true_params = np.array([position_vec[i,0], position_vec[i,1], separation_vec[0,i], angle_vec[0,i]])
    
    # Create target image
    target_image = apply_photon_noise(make_image(true_params, osys)*flux)
    target_image /= np.sum(target_image)
    
    # Default starting params
    #params = np.array([0,0,8,np.pi/2])
    params = 1.01*true_params
    opt = optax.inject_hyperparams(optax.adam)(learning_rate=start_learning_rate)
    opt_state = opt.init(params)
    
    # Do gradient descent
    lr = start_learning_rate
    for j in range(max_iter):
        lr *= 0.95
        #_, opt_update = optax.adam(learning_rate=lr) # i dont think this is actually updating the lr
        grads = jax.grad(compute_loss)(params, osys, target_image)
        #print(grads)
        print('{:.4f}'.format(np.sum(np.abs(grads))), '{:.3e}'.format(lr))
        overall_gtol = np.sum(np.abs(grads))
        
        if overall_gtol < gtol:
            clear_output(wait=True)
            print('Gtol satisfied')
            break
        opt_state.hyperparams['learning_rate'] = lr
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        print(params/true_params)
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
    plt.show()

print(estimated_sep)
print(separation_vec[0])
print(np.abs(estimated_sep - separation_vec)[0,:i+1])

plt.plot(np.abs(estimated_sep - separation_vec)[0,:i+1], 'x')
plt.yscale('log')
plt.subplot(2,2,1)
plt.plot(np.abs(estimated_pos_x - position_vec[:,0])[:i+1], 'x')
plt.title('absolute x pos error')
plt.yscale('log')

plt.subplot(2,2,2)
plt.plot(np.abs(estimated_pos_x - position_vec[:,1])[:i+1], 'x')
plt.title('absolute y pos error')
plt.yscale('log')

plt.subplot(2,2,3)
plt.plot(np.abs(estimated_sep - separation_vec[:,0])[:i+1], 'x')
plt.title('absolute seperation error')
plt.yscale('log')

plt.subplot(2,2,4)
plt.plot(np.abs(estimated_ang - angle_vec[:,0])[:i+1], 'x')
plt.title('absolute angle error')
plt.yscale('log')
plt.show()
