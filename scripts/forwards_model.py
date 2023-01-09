import jax.numpy as np
import matplotlib.pyplot as plt
import jax
import equinox as eqx
import os
import optax
import dLux as dl
import jax
import tqdm.notebook as tqdm


def plot_basis(basis: float, /, shape: tuple = None) -> None:
    """
    Plot a set of polynomial vectors that comprise a
    basis. 
    
    Parameters:
    -----------
    basis: tensor[float]
        The basis vectors arranged as (nvecs, npix, npix).
    rows: tuple[int]
        The number of rows and cols in the plotting grid.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    mpl.rcParams["image.cmap"] = "seismic"
    mpl.rcParams["text.usetex"] = True
    
    number_of_vecs: int = basis.shape[0]
    inches_per_col: int = 3
    inches_per_row: int = 3
    
    # TODO: Implement for odd cases using subfigures to pull 
    #       out the first basis vec into its own centred 
    #       of the same size as the others. 
    if not shape:
        if number_of_vecs % 2 == 0: # is even
            rows: int = 2
            cols: int = number_of_vecs // 2
        else:
            rows: int = 1
            cols: int = number_of_vecs
    else:
        rows: int = shape[0]
        cols: int = shape[1]
    
    if number_of_vecs > rows * cols:
        raise ValueError("Not enough subplots were provided.")
        
    width: int = cols * inches_per_col
    height: int = rows * inches_per_row
    
    min_of_basis: float = basis.min()
    max_of_basis: float = basis.max()
    
    fig = plt.figure(figsize = (width, height))
    axes = fig.subplots(rows, cols)
    
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.axis("off")
    
    for vec in range(number_of_vecs):
        vec_cmap: object = axes.flat[vec].imshow(
            basis[vec][:, ::-1], # Image coordinates
            vmin = min_of_basis, 
            vmax = max_of_basis
        )
        
    fig.subplots_adjust(
        left = 0, 
        right = 1.,
        bottom = .15
    )
    
    col_bar_axis: object = fig.add_axes([0., 0., 1., .08])
        
    col_bar: object = fig.colorbar(
        vec_cmap, 
        cax = col_bar_axis, 
        orientation = "horizontal"
    )
    
    col_bar.ax.tick_params(labelsize = 20)
    col_bar.outline.set_visible(False)
    
    return None


def plot_pupil(pupil: float) -> None:
    """
    Plots an aperture pupil in a nice way. 
    
    Parameters:
    -----------
    pupil: float
        The pupil as an array.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams["text.usetex"]: bool = True
    mpl.rcParams["image.cmap"]: str = "Greys"

    width_in_inches: int = 4

    fig: object = plt.figure(figsize = (width_in_inches, width_in_inches))

    image_axes: object = fig.add_axes([0., 0., .9, .9])
    image_cmap: object = image_axes.imshow(pupil[:, ::-1], vmin = 0., vmax = 1.)

    image_axes.set_xticks([])
    image_axes.set_yticks([])
    image_axes.axis("off")

    cbar_axes: object = fig.add_axes([.95, 0., .05, .9])
    cbar: object = fig.colorbar(image_cmap, cax=cbar_axes)

    cbar.ax.tick_params(labelsize = 20)
    cbar.outline.set_visible(False)
    
    return None


def plot_pupil_with_aberrations(pupil: float, aberrations: float) -> None:
    """
    Plot a pupil that has aberrations.
    
    Parameters:
    -----------
    pupil: float
        The (nearly)-binary array representing the pupil.
    aberrations: float
        The optical path differences. 
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    mpl.rcParams["text.usetex"]: bool = True
    mpl.rcParams["image.cmap"]: str = "seismic"
    
    max_displacement: float = np.max(np.abs(aberrations))
    aberrated_pupil: float = pupil * aberrations.sum(axis = 0)
    width_in_inches: int = 4

    fig: object = plt.figure(figsize = (width_in_inches, width_in_inches))

    image_axes: object = fig.add_axes([0., 0., .9, .9])
    image_cmap: object = image_axes.imshow(
        aberrated_pupil[:, ::-1],
        vmin = - max_displacement,
        vmax = max_displacement
    )

    image_axes.set_xticks([])
    image_axes.set_yticks([])
    image_axes.axis("off")

    cbar_axes: object = fig.add_axes([.95, 0., .05, .9])
    cbar: object = fig.colorbar(image_cmap, cax=cbar_axes)

    cbar.ax.tick_params(labelsize = 20)
    cbar.outline.set_visible(False)
    
    return None 


jax.config.update("jax_enable_x64", True)

mask: float = np.load('../component_models/sidelobes.npy')
npix: int = 1024
detector_npix: int = 100

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
    flux = 1.,
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

toliman_pupil: object = dl.CompoundAperture(
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
)

toliman_aberrations: object = dl.AberratedAperture(
     nolls, 
     coeffs, 
     dl.CircularAperture(
         aperture_diameter / 2.
     )
 )

toliman_mask: object = dl.AddOPD(mask)
toliman_power: object = dl.NormaliseWavefront()

toliman_body: object = dl.AngularMFT(
    detector_npix,
    dl.utils.arcseconds_to_radians(detector_pixel_size)
)

model: object = dl.Instrument(
    optics = toliman,
    sources = [alpha_centauri]
)

toliman: object = dl.Optics(
    layers = [
        wavefront_factory,
        toliman_pupil,
        toliman_aberrations,
        toliman_mask,
        toliman_power,
        toliman_body
    ]
)

comp_model: callable = jax.jit(model.model)
psf: float = model.model()


def plot_psf(psf: float) -> fig:
    fig: object = plt.figure(figsize=(4, 4))
    axes: object = fig.add_axes([0., 0., .95, .95])
    _: list = axes.axis("off")
    caxes: object = fig.add_axes([.95, 0., .05, .95])
    cmap: object = axes.imshow(psf, cmap=plt.cm.inferno)
    cbar: object = fig.colorbar(cmap, cax=caxes)
    _: None = cbar.outline.set_visible(False)


@eqx.filter_jit
@eqx.filter_value_and_grad
def loss(data: float, model: float):
    forward_model: float = model.model()
    return (forward_model - data) ** 2

def apply_photon_noise(image, seed = 0):
    key = jax.random.PRNGKey(seed)
    image_noisy = jax.random.poisson(key = key,lam = image)
    image_noisy /= 1e5  # needed so next line doesn't cause underflow (?) errors for high fluxes
    image_noisy /= np.sum(image_noisy)
    return image_noisy


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
