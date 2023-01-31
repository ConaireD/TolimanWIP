import jax.numpy as np
import jax.random as rng
import os
import shutil

os.environ["TOLIMAN_HOME"] = "tmp"

import toliman.build.background as bg
import toliman.constants as const

RA: int = 0
DEC: int = 1
FLUX: int = 2

def get_background_stars(ra: float, dec: float, rad: float) -> float:
    NUM: int = 100

    def uniform_in_minus_one_to_one(key: int, shape: tuple) -> float:
        return 2.0 * rng.uniform(rng.PRNGKey(key), shape) - 1.0

    ra_sample: float = uniform_in_minus_one_to_one(0, (NUM,)) 
    dec_samples: float = uniform_in_minus_one_to_one(1, (NUM,))
    max_decs: float = np.sqrt(rad ** 2 - ra_sample ** 2)

    return np.array([
            (ra - (rad * ra_sample)),
            (dec -(max_decs * dec_samples)),
            rng.normal(rng.PRNGKey(2), (NUM,)),
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
    # Arrange
    bg_stars: float = get_background_stars(3.0, 3.0, 2.0)

    # Act
    win_bg_stars: float = bg.window_background_stars(bg_stars, np.sqrt(2.0)) 

    # Assert
    assert (win_bg_stars[RA] <= np.sqrt(2.0)).all()
    assert (win_bg_stars[DEC] <= np.sqrt(2.0)).all()

def test_window_background_stars_has_correct_shape():
    # Arrange
    bg_stars: float = get_background_stars(3.0, 3.0, 2.0)

    # Act
    win_bg_stars: float = bg.window_background_stars(bg_stars, np.sqrt(2.0)) 

    # Assert
    assert win_bg_stars.shape[0] == 3

def test_flux_relative_to_alpha_cen_has_correct_shape():
    # Arrange
    bg_stars: float = get_background_stars(3.0, 3.0, 2.0)

    # Act
    rel_bg_stars: float = bg.flux_relative_to_alpha_cen(bg_stars)

    # Assert
    assert rel_bg_stars.shape[0] == 3

def test_save_background_stars_creates_file():
    # Arrange
    if os.path.isdir("tmp"):
        shutil.rmtree("tmp")

    os.mkdir("tmp")
    const.set_const("TOLIMAN_HOME", "tmp")

    bg_stars: float = get_background_stars(3.0, 3.0, 2.0)

    # Act
    bg.save_background_stars(bg_stars)

    # Assert
    assert os.path.isfile("tmp/background.csv")

    # Clean Up
    shutil.rmtree("tmp")

def test_save_background_stars_has_correct_headings():
    # Arrange
    if os.path.isdir("tmp"):
        shutil.rmtree("tmp")

    os.mkdir("tmp")
    const.set_const("TOLIMAN_HOME", "tmp")

    bg_stars: float = get_background_stars(3.0, 3.0, 2.0)

    # Act
    bg.save_background_stars(bg_stars)

    # Assert
    with open("tmp/background.csv", "r") as file:
        headings: str = next(file).strip().split(",")
        assert headings[0] == "ra"
        assert headings[1] == "dec"
        assert headings[2] == "rel_flux"

    # Clean Up
    shutil.rmtree("tmp")

def test_save_background_stars_has_correct_number_of_lines():
    # Arrange
    if os.path.isdir("tmp"):
        shutil.rmtree("tmp")

    os.mkdir("tmp")
    const.set_const("TOLIMAN_HOME", "tmp")

    bg_stars: float = get_background_stars(3.0, 3.0, 2.0)

    # Act
    bg.save_background_stars(bg_stars)

    # Assert
    with open("tmp/background.csv", "r") as file:
        assert len(file.readlines()) == bg_stars[0].size

    # Clean Up
    shutil.rmtree("tmp")
