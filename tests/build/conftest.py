import os
import shutil
import pytest
import toliman.constants as const

FILTER_MIN_WAVELENGTH: float = const.get_const_as_type("FILTER_MIN_WAVELENGTH", float)
FILTER_MAX_WAVELENGTH: float = const.get_const_as_type("FILTER_MAX_WAVELENGTH", float)

@pytest.fixture
def create_fake_phoenix_installation(
        root: str, 
        full: bool, 
    ) -> None:
    numbers: list = [5200, 5300, 5700, 5800]
    phoenix: str = "/".join([root, "grid", "phoenix"])
    phoenixm00: str = "/".join([phoenix, "phoenixm00"])
    phoenixp03: str = "/".join([phoenix, "phoenixp03"])
        
    os.makedirs(phoenixm00)
    os.makedirs(phoenixp03)

    open("{}/catalog.fits".format(phoenix), "w").close()
    for num in numbers:
        open("{}/phoenixm00_{}.fits".format(phoenixm00, num), "w").close()
        if full:
            open("{}/phoenixp03_{}.fits".format(phoenixp03, num), "w").close()

    yield 

    if os.path.isdir(root): shutil.rmtree(root)

@pytest.fixture
def create_fake_mask_installation(root: str) -> None:
    os.makedirs(root)
    open("{}/mask.npy".format(root)).close()
    
@pytest.fixture
def create_fake_background_installation(root: str) -> None:
    os.makedirs(root)
    open("{}/background.npy".format(root)).close()

@pytest.fixture(autouse=True)
def remove_installation(root: str) -> None:
    if os.path.isdir(root): shutil.rmtree(root)
