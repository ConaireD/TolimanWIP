import os
import shutil
import pytest
import toliman.constants as const
import typing 
import 

class fixture(typing.Generic[typing.TypeVar("T")]): pass

@pytest.fixture
def make_fake_spectra(min_: float, max_: float) -> None:
    shape: int = 100
    return jax.numpy.array([
            jax.numpy.linspace(min_, max_, shape, dtype = float),
            jax.random.norm(jax.random.PRNGKey(0), (shape,), dtype = float),
            jax.random.norm(jax.random.PRNGKey(1), (shape,), dtype = float),
        ], dtype = float)

@pytest.fixture
def list_phoenix_dirs(root: str) -> list:
    phoenix: str = "/".join([root, "grid", "phoenix"])
    phoenixm00: str = "/".join([phoenix, "phoenixm00"])
    phoenixp03: str = "/".join([phoenix, "phoenixp03"])
    return [phoenixm00, phoenixp03]

@pytest.fixture
def list_phoenix_files(root: str) -> list:
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

