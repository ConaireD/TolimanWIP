import toliman.math as math
import jax.numpy as np

def test_normalise_in_range():
    array: float = np.arange(10)
    narray: float = math.normalise(array)
    assert (0.0 <= narray).all() and (narray <= 1.0).all()
