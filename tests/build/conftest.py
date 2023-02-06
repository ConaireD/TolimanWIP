import os
import shutil
import pytest
import toliman.constants as const
import typing 
import jax 

class fixture(typing.Generic[typing.TypeVar("T")]): pass

@pytest.fixture
def make_fake_background_stars(ra: float, dec: float, rad: float) -> float:
    """
    Create a phony array representing background stars.

    Parameters
    ----------
    ra: float, deg
        The right ascension of the stars.
    dec: float, deg
        The declination of the stars.
    rad: float, deg
        The radius to place the stars in.

    Returns
    -------
    stars: float, [deg, deg, W/m/m]
        An array based representation of the stars. The convention is 
        [RA, DEC, FLUX].
    """
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
    
@pytest.fixture
def make_fake_spectra(min_: float, max_: float) -> None:
    """
    Create a phony spectra represented as an array.

    Parameters
    ----------
    min_: float, meters
        The smallest wavelengths.
    min_: float, meters
        The largest wavelengths.

    Returns
    -------
    spectra: float, [m, W/m/m, W/m/m]
        The spectra of the two stars. The common wavelengths is the 
        first row and the flux occupy the next two rows.
    """
    shape: int = 100
    return jax.numpy.array([
            jax.numpy.linspace(min_, max_, shape, dtype = float),
            jax.random.normal(jax.random.PRNGKey(0), (shape,), dtype = float),
            jax.random.normal(jax.random.PRNGKey(1), (shape,), dtype = float),
        ], dtype = float)

@pytest.fixture
def list_phoenix_dirs(root: str) -> list:
    """
    List the phoenix directories.

    Parameters
    ----------
    root: str
        The phoenix root directory.

    Returns
    -------
    dirs: list[str]
        The directories containing the phoenix files.
    """
    phoenix: str = "/".join([root, "grid", "phoenix"])
    phoenixm00: str = "/".join([phoenix, "phoenixm00"])
    phoenixp03: str = "/".join([phoenix, "phoenixp03"])
    return [phoenixm00, phoenixp03]

@pytest.fixture
def list_phoenix_files(root: str) -> list:
    """
    List the phoenix files.

    Parameters
    ----------
    root: str
        The phoenix root directory.

    Results
    -------
    files: list[str]
        The phoenix files.
    """
    phoenix: str = "/".join([root, "grid", "phoenix"])
    phoenixm00: str = "/".join([phoenix, "phoenixm00"])
    phoenixp03: str = "/".join([phoenix, "phoenixp03"])
    numbers: list = [5200, 5300, 5700, 5800] 
    return ["{}/catalog.fits".format(phoenix)] +\
        ["{}/phoenixm00_{}.fits".format(phoenixm00, num) for num in numbers] +\
        ["{}/phoenixp03_{}.fits".format(phoenixp03, num) for num in numbers]

@pytest.fixture
def create_fake_phoenix_dirs(
        root: str, 
        full: bool, 
        remove_installation: fixture[None],
        list_phoenix_dirs: fixture[list],
    ) -> None:
    if full: 
        for pdir in list_phoenix_dirs: 
            os.makedirs(pdir)
    else:
        os.makedirs(list_phoenix_dirs[0])
    
@pytest.fixture
def create_fake_phoenix_installation(
        root: str, 
        full: bool, 
        remove_installation: fixture[None], 
        create_fake_phoenix_dirs: fixture[None],
    ) -> None:
    phoenix: str = "/".join([root, "grid", "phoenix"])
    phoenixm00: str = "/".join([phoenix, "phoenixm00"])
    phoenixp03: str = "/".join([phoenix, "phoenixp03"])
    numbers: list = [5200, 5300, 5700, 5800]
    open("{}/catalog.fits".format(phoenix), "w").close()
    for num in numbers:
        open("{}/phoenixm00_{}.fits".format(phoenixm00, num), "w").close()
        if full:
            open("{}/phoenixp03_{}.fits".format(phoenixp03, num), "w").close()
            
@pytest.fixture
def create_fake_mask_installation(root: str) -> None:
    remove_installation(root)
    os.makedirs(root)
    open("{}/mask.npy".format(root)).close()
    
@pytest.fixture
def create_fake_background_installation(root: str) -> None:
    remove_installation(root)
    os.makedirs(root)
    open("{}/background.npy".format(root)).close()

@pytest.fixture
def remove_installation(root: str) -> None:
    if os.path.isdir(root): shutil.rmtree(root)
    yield
    if os.path.isdir(root): shutil.rmtree(root)

@pytest.fixture
def setup_and_teardown_phoenix_environ() -> None:
    if os.environ.get("PYSYN_CDBS"): os.environ.pop("PYSYN_CDBS")
    yield
    if os.environ.get("PYSYN_CDBS"): os.environ.pop("PYSYN_CDBS")

