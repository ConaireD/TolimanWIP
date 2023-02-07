import jax.numpy as np
import jax.random as rng
import os
import shutil
import pytest

os.environ["TOLIMAN_HOME"] = "tmp"

import toliman.build.background as bg
import toliman.constants as const
import typing

RA: int = 0
DEC: int = 1
FLUX: int = 2
ROOT: str = "tmp"
        
class fixture(typing.Generic[typing.TypeVar("T")]): pass

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
    assert (win_bg_stars[(RA, DEC), :] <= window).all()

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

@pytest.mark.parametrize("ra", [3.0])
@pytest.mark.parametrize("dec", [3.0])
@pytest.mark.parametrize("rad", [2.0])
@pytest.mark.parametrize("root", [ROOT])
def test_save_background_stars_creates_file(
        root: str,
        remove_installation: fixture[None],
        make_fake_background_stars: fixture[float],
    ) -> None:
    """
    Does bg.save_background_stars create the correct file?

    Fixtures
    --------
    remove_installation: fixture[None],
       Ensure there is no installation before and after the test. 
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
    root: str = ROOT
        The directory to background stars to.
    """
    bg.save_background_stars(make_fake_background_stars, root)
    assert os.path.isfile("{}/background.csv".format(root))

@pytest.mark.parametrize("ra", [3.0])
@pytest.mark.parametrize("dec", [3.0])
@pytest.mark.parametrize("rad", [2.0])
@pytest.mark.parametrize("root", [ROOT])
def test_save_background_stars_has_correct_headings(
        root: str,
        remove_installation: fixture[None],
        make_fake_background_stars: fixture[float],
    ) -> None:
    """
    Does bg.save_background_stars create the correct columns?

    Fixtures
    --------
    remove_installation: fixture[None],
       Ensure there is no installation before and after the test. 
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
    root: str = ROOT
        The directory to background stars to.
    """
    bg.save_background_stars(make_fake_background_stars, root)
    with open("{}/background.csv".format(root), "r") as file:
        headings: str = next(file).strip().split(",")
        assert headings[0] == "ra"
        assert headings[1] == "dec"
        assert headings[2] == "rel_flux"

@pytest.mark.parametrize("ra", [3.0])
@pytest.mark.parametrize("dec", [3.0])
@pytest.mark.parametrize("rad", [2.0])
@pytest.mark.parametrize("root", [ROOT])
def test_save_background_stars_has_correct_number_of_lines(
        root: str,
        remove_installation: fixture[None],
        make_fake_background_stars: fixture[float],
    ):
    """
    Does bg.save_background_stars create the correct volume of data?

    Fixtures
    --------
    remove_installation: fixture[None],
       Ensure there is no installation before and after the test. 
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
    root: str = ROOT
        The directory to background stars to. Indirectly parametrizes 
        remove_installation.
    """
    bg.save_background_stars(make_fake_background_stars, root)
    with open("{}/background.csv".format(root), "r") as file:
        assert len(file.readlines()) == make_fake_background_stars[0].size + 1

@pytest.mark.parametrize("root", [ROOT])
def test_are_background_stars_installed_when_installed(
        root: str,
        remove_installation: fixture[None],
        create_fake_background_installation: fixture[None],
    ) -> None:
    """
    Does bg.are_background_stars_installed detect an existing installation?

    Fixtures
    --------
    remove_installation: fixture[None],
        Ensures that there is no installation before and after the test.
    create_fake_background_installation: fixture[None],
        Quickly makes a false installation.

    Parameters
    ----------
    root: str = ROOT
        The directory of installation. Directly parametrizes 
        bg.are_background_stars_installed. Indirectly parametrizes
        remove_installation and create_fake_background_installation.
    """
    assert bg.are_background_stars_installed(root)

@pytest.mark.parametrize("root", [ROOT])
def test_are_background_stars_installed_when_not_installed(
        root: str,
        remove_installation: fixture[None],
    ) -> None:
    """
    Does bg.are_background_stars_installed detect no existing installation?

    Fixtures
    --------
    remove_installation: fixture[None],
        Ensures that there is no installation before and after the test.

    Parameters
    ----------
    root: str = ROOT
        The directory of installation. Directly parametrizes 
        bg.are_background_stars_installed. Indirectly parametrizes
        remove_installation. 
    """
    assert not bg.are_background_stars_installed(root)
