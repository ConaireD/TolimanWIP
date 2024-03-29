import toliman.math as math
import jax.numpy as np
import jax.random as jr
import jax.lax as jl
import pytest 
import typing

class fixture(typing.Generic[typing.TypeVar("T")]): pass

# TODO: Paramatrize by shape.
# TODO: Use jax.random
def test_normalise_in_range():
    array: float = np.arange(10)
    narray: float = math.normalise(array)
    assert (0.0 <= narray).all() and (narray <= 1.0).all()

@pytest.mark.parametrize("shape", [100, 256])
@pytest.mark.parametrize("m", [1, 2, 5, 10])
def test_downsample_square_grid_has_correct_shape_when_valid(
        m: int,
        shape: int
    ) -> None:
    """
    Does math.downsample_on_square_grid downsample?

    Parameters
    ----------
    m: int
        The amount to downsample the array by.
    shape: int
        The initial shape size of the array.
    """
    array: float = np.zeros((shape, shape), dtype = float)
    resampled: float = math.downsample_square_grid(array, m)
    assert resampled.shape == (shape // m, shape // m)

@pytest.mark.parametrize("shape", [100, 256])
@pytest.mark.parametrize("m", [1, 2, 5, 10])
def test_downsample_square_grid_in_range(
        m: int,
        shape: int
    ) -> None:
    """
    Does math.downsample_on_square_grid average?

    The average should be less than or equal to the largest value and
    greater than or equal to the smallest value. 

    Parameters
    ----------
    m: int
        The amount to downsample the array by.
    shape: int
        The initial shape size of the array.
    """
    array: float = jr.normal(jr.PRNGKey(0), (shape, shape), dtype = float)
    resampled: float = math.downsample_square_grid(array, m)
    assert (array.max() >= resampled).all() and (array.min() <= resampled).all()

@pytest.mark.parametrize("shape", [100, 256])
@pytest.mark.parametrize("m", [1, 2, 5, 10])
def test_downsample_along_axis_has_correct_shape_when_valid(
        m: int,
        shape: int
    ) -> None:
    """
    Does math.downsample_on_square_grid downsample?

    Parameters
    ----------
    m: int
        The amount to downsample the array by.
    shape: int
        The initial shape size of the array.
    """
    array: float = np.zeros((shape, 2), dtype = float)
    resampled: float = math.downsample_along_axis(array, m, axis = 0)
    assert resampled.shape == (shape // m, 2)

@pytest.mark.parametrize("shape", [100, 256])
@pytest.mark.parametrize("m", [1, 2, 5, 10])
def test_downsample_along_axis_in_range(
        m: int,
        shape: int,
    ):
    """
    The average should be less than or equal to the largest value and
    greater than or equal to the smallest value. 

    Parameters
    ----------
    m: int
        The amount to downsample the array by.
    shape: int
        The initial shape size of the array.
    """
    array: float = jr.normal(jr.PRNGKey(0), (shape, 2), dtype = float)
    resampled: float = math.downsample_along_axis(array, m)
    maximums: float = np.max(array, axis = 0)
    minimums: float = np.min(array, axis = 0)
    assert (maximums >= resampled).all() and (minimums <= resampled).all()

@pytest.mark.parametrize("shape", [64, 128])
def test_photon_noise_is_integer(
        make_airy_psf: fixture[float],
    ) -> None:
    """
    Does math.photon_noise have an integer ouput?

    Fixtures
    --------
    make_airy_psf: fixture[None]
        Generate an airy pattern.

    Parameters    plt.imshow(edge)
    plt.colorbar()
    plt.show()
    ----------
    shape: int
        Pixels in the psf. Indirectly parametrizes make_airy_psf.
    """
    noisy_psf: int = math.photon_noise(make_airy_psf)
    assert (noisy_psf.dtype == np.int32) or (noisy_psf.dtype == np.int64) 

@pytest.mark.parametrize("shape", [64, 128])
def test_photon_noise_is_correct_shape(
        make_airy_psf: fixture[float]
    ) -> None:
    """
    Does the noisy psf have the same shape as the psf?

    Fixtures
    --------
    make_airy_psf: fixture[None]
        Generate an airy pattern.

    Parameters
    ----------
    shape: int
        Pixels in the psf. Indirectly parametrizes make_airy_psf.
    """
    noisy_psf: int = math.photon_noise(make_airy_psf)
    assert noisy_psf.shape == make_airy_psf.shape

@pytest.mark.parametrize("factor", [2, 5, 10])
@pytest.mark.parametrize("shape", [64])
def test_photon_noise_scales_with_psf(
        factor: float,
        make_airy_psf: fixture[float],
    ) -> None:
    """
    Does the amount of photon noise scale with the psf?

    Fixtures
    --------
    make_airy_psf: fixture[None]
        Generate an airy pattern.

    Parameters
    ----------
    shape: int
        Pixels in the psf. Indirectly parametrizes make_airy_psf.
    factor: float
        Multiply the psf by this.
    """
    sum_of_psf: float = np.sum(math.photon_noise(make_airy_psf))
    sum_of_noisy_psf: float = np.sum(math.photon_noise(factor * make_airy_psf))
    assert sum_of_noisy_psf > sum_of_psf

@pytest.mark.parametrize("shape", [64])
def test_airy_photon_noise_on_disk(
        shape: int,
        coordinates: fixture[float],
        make_airy_psf: fixture[float],
    ) -> None:
    """
    Does math.photon_noise approximately retain the power distribution?

    Fixtures
    --------
    coordinates: fixture[float]
        The coordinates of the psf.
    make_airy_psf: fixture[None]
        Generate an airy pattern.

    Parameters
    ----------
    shape: int
        Pixels in the psf. Indirectly parametrizes make_airy_psf.
    """
    psf: float = math.photon_noise(make_airy_psf)
    power: float = np.sum(psf)
    centre: float = jl.lt(jl.abs(coordinates), shape / 8.)
    peak: float = np.where(centre, psf, 0.)
    peak_power: float = np.sum(peak)
    assert peak_power > 0.8 * power

@pytest.mark.parametrize("scale", [1.0, 2.0])
@pytest.mark.parametrize("shape", [64, 128, 256])
def test_latent_detector_noise_is_uniform(
        shape: int, 
        scale: float,
        coordinates: fixture[float],
    ) -> None:
    """
    Is the latent detector noise uniform?

    Fixtures
    --------
    coordinates: fixture[float]
        The coordinates of the psf.

    Parameters
    ----------
    shape: int
        The number of pixels along each side. Indirectly parametrizes
        coordinates.
    scale: float
        The scaling factor of the detector noise.
    """
    noise: float = np.abs(math.latent_detector_noise(scale, (shape, shape)))
    quadrants: int = np.array([
            (coordinates[0] > 0.0) & (coordinates[1] > 0.0),
            (coordinates[0] < 0.0) & (coordinates[1] > 0.0),
            (coordinates[0] < 0.0) & (coordinates[1] < 0.0),
            (coordinates[0] > 0.0) & (coordinates[1] < 0.0),
        ], dtype = int)
    
    noise_in_quadrants: float = np.where(quadrants, noise, 0.)
    power_in_quadrants: float = np.sum(noise_in_quadrants, axis = (1, 2))
    average_power: float = np.sum(noise) / 4.0

    assert ((np.abs(power_in_quadrants - average_power) / average_power) < 0.2).all()

@pytest.mark.parametrize("threshold", [0.1, 0.05])
@pytest.mark.parametrize("shape", [64, 128, 256])
def test_pixel_response_is_uniform(
        shape: int, 
        threshold: float,
        coordinates: fixture[float],
    ) -> None:
    """
    Is the pixel response uniform?

    Fixtures
    --------
    coordinates: fixture[float]
        The coordinates of the psf.

    Parameters
    ----------
    shape: int
        The number of pixels along each side. Indirectly parametrizes
        coordinates.
    scale: float
        The scaling factor of the detector noise.
    """
    noise: float = np.abs(math.pixel_response((shape, shape), threshold))
    quadrants: int = np.array([
            (coordinates[0] > 0.0) & (coordinates[1] > 0.0),
            (coordinates[0] < 0.0) & (coordinates[1] > 0.0),
            (coordinates[0] < 0.0) & (coordinates[1] < 0.0),
            (coordinates[0] > 0.0) & (coordinates[1] < 0.0),
        ], dtype = int)
    
    noise_in_quadrants: float = np.where(quadrants, noise, 0.)
    power_in_quadrants: float = np.sum(noise_in_quadrants, axis = (1, 2))
    average_power: float = np.sum(noise) / 4.0

    assert ((np.abs(power_in_quadrants - average_power) / average_power) < 0.2).all()
