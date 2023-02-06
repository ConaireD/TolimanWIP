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
ROOT: str = "tmp"
BG: str = "tmp/background.csv"
        
def test_load_background_stars_has_correct_shape() -> None:
    """
    Does bg.load_background_stars have the correct amount of data?
    """
    bg_stars: float = bg.load_background_stars(3.0, 3.0, 2.0)
    assert bg_stars.shape[0] == 3

def test_load_background_stars_within_cone() -> None:
    """
    Does bg.load_background_stars load stars in a cone?

    When manually reviewing the output of bg.load_background_stars it
    was found that the cone had a certain "tolerance" associated with 
    it. We have given a rather lenient tolerance for the tests.
    """
    # Arrange
    rad: float = 2.0
    tol: float = 0.01
    bg_stars: float = bg.load_background_stars(3.0, 3.0, rad)

    # Act/Assert
    hypot: callable = lambda x: np.hypot(x[0], x[1])
    hypots: float = np.apply_along_axis(hypot, 0, bg_stars[(0, 1), :])

    assert (hypots <= rad + tol).all()

@pytest.mark.parametrize("ra", [3.0])
@pytest.mark.parametrize("dec", [3.0])
@pytest.mark.parametrize("rad", [2.0])
def test_window_background_stars_in_range(
        make_fake_background_stars: fixture[float],
    ) -> None:
    """
    Does bg.window_background_stars enclose the correct shape?

    Fixtures
    --------
    make_fake_background_stars: fixture[float],
        Quickly generate an array of imaginary background stars.

    Parameters
    ----------
    ra: float = 3.0, deg
        The right ascension centre of the background stars. Indirectly 
        parametrizes make_fake_background_stars.
    dec: float = 3.0, deg
        The declination centre of the background stars. Indirectly parametrizes
        make_fake_background_stars.
    rad: float = 2.0, deg
        The radius of the search window. Indirectly parametrizes 
        make_fake_background_stars.
    """
    window: float = np.sqrt(2.0)
    win_bg_stars: float = bg.window_background_stars(
        make_fake_background_stars, 
        window,
    ) 
    assert (win_bg_stars[(RA, DEC)] <= window).all()

@pytest.mark.parametrize("ra", [3.0])
@pytest.mark.parametrize("dec", [3.0])
@pytest.mark.parametrize("rad", [2.0])
def test_window_background_stars_has_correct_shape(
        make_fake_background_stars: fixture[None],
    ) -> None:
    """
    Does bg.window_background_stars keep the length of the leading dimension?

    Fixtures
    --------
    make_fake_background_stars: fixture[float],
        Quickly generate an array of imaginary background stars.

    Parameters
    ----------
    ra: float = 3.0, deg
        The right ascension centre of the background stars. Indirectly 
        parametrizes make_fake_background_stars.
    dec: float = 3.0, deg
        The declination centre of the background stars. Indirectly parametrizes
        make_fake_background_stars.
    rad: float = 2.0, deg
        The radius of the search window. Indirectly parametrizes 
        make_fake_background_stars.
    """
    window: float = np.sqrt(2.0)
    win_bg_stars: float = bg.window_background_stars(make_fake_background_stars, window) 
    assert win_bg_stars.shape[0] == 3

@pytest.mark.parametrize("ra", [3.0])
@pytest.mark.parametrize("dec", [3.0])
@pytest.mark.parametrize("rad", [2.0])
def test_flux_relative_to_alpha_cen_has_correct_shape(
        make_fake_background_stars: fixture[float],
    ) -> None:
    """
    Does bg.flux_relative_to_alpha_cen keep the length of the leading dimension?

    Fixtures
    --------
    make_fake_background_stars: fixture[float],
        Quickly generate an array of imaginary background stars.

    Parameters
    ----------
    ra: float = 3.0, deg
        The right ascension centre of the background stars. Indirectly 
        parametrizes make_fake_background_stars.
    dec: float = 3.0, deg
        The declination centre of the background stars. Indirectly parametrizes
        make_fake_background_stars.
    rad: float = 2.0, deg
        The radius of the search window. Indirectly parametrizes 
        make_fake_background_stars.
    """
    rel_bg_stars: float = bg.flux_relative_to_alpha_cen(make_fake_background_stars)
    assert rel_bg_stars.shape[0] == 3

def test_save_background_stars_creates_file():
    # Arrange
    if os.path.isdir(ROOT):
        shutil.rmtree(ROOT)

    os.mkdir(ROOT)
    bg_stars: float = get_background_stars(3.0, 3.0, 2.0)

    # Act
    bg.save_background_stars(bg_stars, ROOT)

    # Assert
    assert os.path.isfile(BG)

    # Clean Up
    shutil.rmtree(ROOT)

def test_save_background_stars_has_correct_headings():
    # Arrange
    if os.path.isdir(ROOT):
        shutil.rmtree(ROOT)

    os.mkdir(ROOT)
    bg_stars: float = get_background_stars(3.0, 3.0, 2.0)

    # Act
    bg.save_background_stars(bg_stars, ROOT)

    # Assert
    with open(BG, "r") as file:
        headings: str = next(file).strip().split(",")
        assert headings[0] == "ra"
        assert headings[1] == "dec"
        assert headings[2] == "rel_flux"

    # Clean Up
    shutil.rmtree(ROOT)

def test_save_background_stars_has_correct_number_of_lines():
    # Arrange
    if os.path.isdir(ROOT):
        shutil.rmtree(ROOT)

    os.mkdir(ROOT)
    bg_stars: float = get_background_stars(3.0, 3.0, 2.0)

    # Act
    bg.save_background_stars(bg_stars, ROOT)

    # Assert
    with open(BG, "r") as file:
        assert len(file.readlines()) == bg_stars[0].size + 1

    # Clean Up
    shutil.rmtree(ROOT)

def test_are_background_stars_installed_when_installed() -> None:
    # Arrange
    if os.path.isdir(ROOT):
        shutil.rmtree(ROOT)

    os.mkdir(ROOT)
    open(BG, "w").close()

    # Act/Assert
    assert bg.are_background_stars_installed(ROOT)

def test_are_background_stars_installed_when_not_installed() -> None:
    # Arrange
    if os.path.isdir(ROOT):
        shutil.rmtree(ROOT)

    os.mkdir(ROOT)

    # Act/Assert
    assert not bg.are_background_stars_installed(ROOT)
