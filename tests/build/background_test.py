import toliman.build.background as bg
import toliman.constants as const
import jax.numpy as np
import jax.random as rng

def get_background_stars(ra: float, dec: float, rad: float) -> float:
    NUM: int = 100

    def uniform_in_minus_one_to_one(key: int, shape: tuple) -> float:
        return 2.0 * rng.uniform(rng.PRNGKey(key), shape) - 1.0

    ra_sample: float = uniform_in_minus_one_to_one(0, NUM) 
    dec_samples: float = uniform_in_minus_one_to_one(1, NUM)
    max_decs: float = np.sqrt(rad ** 2 - ra_sample ** 2)

    return np.array([
            (ra - (rad * ra_sample)),
            (dec -(max_decs * dec_samples)),
            rng.norm(rng.PRNGKey(2), NUM),
        ], dtype = float)
        
def test_load_background_stars_has_correct_shape():
    # Arrange
    bg_stars: float = bg.load_background_stars(3.0, 3.0, 2.0)

    # Act/Assert
    assert bg_stars.shape[0] == 3

def test_load_background_stars_within_cone():
    # Arrange
    rad: float = 2.0
    tol: float = 0.01
    bg_stars: float = bg.load_background_stars(3.0, 3.0, rad)

    print(bg_stars)

    # Act/Assert
    hypot: callable = lambda x: np.hypot(x[0], x[1])
    hypots: float = np.apply_along_axis(hypot, 0, bg_stars[(0, 1), :])

    assert (hypots <= rad + tol).all()

def test_window_background_stars_in_range():
    
