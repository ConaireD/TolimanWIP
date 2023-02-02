import os
import shutil
import pytest
import toliman.constants as const
import typing 

class fixture(typing.Generic[typing.TypeVar("T")]): pass

FILTER_MIN_WAVELENGTH: float = const.get_const_as_type("FILTER_MIN_WAVELENGTH", float)
FILTER_MAX_WAVELENGTH: float = const.get_const_as_type("FILTER_MAX_WAVELENGTH", float)

class PhoenixTree(object):
    _iter: int
    phoenix: str
    phoenixm00: str
    phoenixp03: str
    def __init__(self: object, root: str) -> object:
        self._iter: int = 0
        self.phoenix: str = "/".join([root, "grid", "phoenix"])
        self.phoenixm00: str = "/".join([self.phoenix, "phoenixm00"])
        self.phoenixp03: str = "/".join([self.phoenix, "phoenixp03"])
    def __iter__(self: object) -> iter:
        self._iter: int = 0
        return self
    def __next__(self: object) -> str:
        if self._iter > 1:
            raise StopIteration
        elif self._iter == 1:
            self._iter: int = 2
            return self.phoenixp03
        elif self._iter == 0:
            self._iter: int = 1
            return self.phoenixm00
        else:
            raise ValueError("How did we get here?")

    
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
