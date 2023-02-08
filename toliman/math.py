import jax.numpy as np
import jax

__author__ = [
    "Jordan Dennis",
    "Louis Desdoigts",
]

__all__ = [
    "downsample_square_grid", 
    "downsample_along_axis", 
    "simulate_data",
    "pixel_response",
    "photon_noise",
    "latent_detector_noise",
    "normalise",
]

def downsample_square_grid(arr: float, resample_by: int) -> float:
    """
    Resample a square array by a factor of `m`.

    Parameters
    ----------
    arr: float
        An `NxN` array.
    m: float
        The factor to downsample by so that the final shape is `(N/m)x(N/m)`.
        This implies that `N % m == 0`.

    Examples
    --------
    ```python
    >>> import jax.numpy as np
    >>> up_arr: float = np.ones((1024, 1024), dtype=float)
    >>> down_arr: float = _downsample(arr, 4)
    >>> down_arr.shape
    (256, 256)
    ```
    """
    shape_in: int = arr.shape[0]
    shape_out: int = shape_in // resample_by
    keep_from_left: int = shape_in - shape_in % resample_by
    kept_array: float = arr[:keep_from_left, :keep_from_left]
    kept_shape: int = kept_array.shape[0]

    shape_for_first_sum: tuple = (kept_shape * shape_out, resample_by)
    shape_for_second_sum: tuple = (shape_out * shape_out, resample_by) 
    sum_on_first_ax: float = kept_array.reshape(shape_for_first_sum).sum(1)
    one_ax_summed: float = sum_on_first_ax.reshape(kept_shape, shape_out).T
    sum_on_second_ax: float = one_ax_summed.reshape(shape_for_second_sum).sum(1)
    summed: float = sum_on_second_ax.reshape(shape_out, shape_out).T

    return summed / resample_by / resample_by

def downsample_along_axis(arr: float, m: int, axis: int = 0) -> float:
    """
    Resampling an array by averaging along a particular dimension.

    Parameters
    ----------
    arr: float
        The array to resample.
    m: int
        The factor by which to downsample the axis.
    axis: int = 0
        The axis to resample.

    Returns
    -------
    arr: float
        The resampled array
    """
    shape: tuple = arr.shape
    n: int = shape[axis]
    out: int = n // m
    new: tuple = tuple([out if i == axis else dim for i, dim in enumerate(shape)] + [m])
    return arr.reshape(new).sum(-1) / m

def simulate_data(model: object, scale: float) -> float:
    """
    Simulate some fake sata for comparison.

    Parameters
    ----------
    model: object
        A model of the toliman. Should inherit from `dl.Instrument` or
        be an instance.
    scale: float
        How noisy is the detector?

    Returns
    -------
    data: float, photons
        A noisy psf.
    """
    psf: float = model.model()
    noisy_psf: float = photon_noise(psf)
    noisy_image: float = noisy_psf + latent_detector_noise(scale, psf.shape)
    return noisy_image

def pixel_response(shape: float, threshold: float, seed: int = 1) -> float:
    """
    Simulate pixel reponses.

    Parameters
    ----------
    shape: tuple[int]
        The array shape to populate with a random pixel response.
    threshold: float
        How far from 1. does the pixel response typically vary.
    seed: int = 1
        The seed of the random generation.

    Returns
    -------
    pixel_response: float
        An array of the pixel responses.
    """
    key: object = jax.random.PRNGKey(seed)
    return 1.0 + threshold * jax.random.normal(key, shape)

def photon_noise(psf: float, seed: int = 0) -> float:
    """
    Simulate photon noise.

    Parameters
    ----------
    psf: float
        The psf on which to add photon noise.
    seed: int = 1
        The seed of the random generation.

    Returns
    -------
    photon_noise: float
        A noisy psf.
    """
    key = jax.random.PRNGKey(seed)
    return jax.random.poisson(key, psf)

def latent_detector_noise(scale: float, shape: float, seed: int = 0) -> float:
    """
    Simulate some gaussian latent noise.

    Parameters
    ----------
    scale: float, photons
        The standard deviation of the gaussian in photons.
    shape: tuple
        The shape of the array to generate the noise on.
    seed: int = 0
        The seed of the random generation.

    Returns
    -------
    det_noise: float, photons
        The an additional noise source from the detector.
    """
    key: object = jax.random.PRNGKey(seed)
    return scale * jax.random.normal(key, shape)

def normalise(arr: float) -> float:
    """
    Rescale and array onto [0, 1].

    Parameters
    ----------
    arr: float
        Any array.

    Returns
    -------
    arr: float
        An array of floating point numbers over the range [0, 1].
    """
    return (arr - arr.min()) / arr.ptp()
        
def angstrom_to_m(angstrom: float) -> float:
    """
    Convert an array that is in angstrom to meters.

    Parameters
    ----------
    angstrom: float, angstrom
        An array of measurements.

    Returns
    -------
    meters: float, meters
        An array of measurements.
    """
    m_per_angstrom: float = 1e-10
    return m_per_angstrom * angstrom
