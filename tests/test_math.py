import toliman.math as math
import jax.numpy as np
import jax.random as jr
import pytest 

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


#def test_downsample_along_axis
#def test_photon_noise
#def test_latent_detector_noise
#def test_simulate_data
#def test_pixel_response
