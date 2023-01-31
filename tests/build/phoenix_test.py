import pytest
import shutil
import os
import toliman.build.phoenix as phoenix
import toliman.build.paths as paths
import toliman.constants as const
import jax.numpy as np
import jax.random as random

# TODO: Make into fixtures
ASSETS: str = "tmp"
PHOENIX: str = "{}/grid/phoenix".format(ASSETS)
PHOENIXS: str = ["phoenixm00", "phoenixp03"]
NUMBERS: list = [5200, 5300, 5700, 5900]
PHOENIX_FILES: list = ["{}/catalog.fits".format(PHOENIX)] + [
    "{}/{}/{}_{}.fits".format(PHOENIX, phoe, phoe, num) 
        for num in NUMBERS 
        for phoe in PHOENIXS
]

def make_phoenix_root_directory() -> None:
    if not os.path.exists(PHOENIX):
        for path in paths.accumulate(PHOENIX.split("/")):
            if not os.path.exists(path):
                os.mkdir(path)

def make_phoenix_type_directory(phoenix: str) -> None:
    if not os.path.exists(PHOENIX):
        make_phoenix_root_directory()
    path: str = paths.concat([PHOENIX, phoenix])
    if not os.path.exists(path):
        os.mkdir(path)

def make_phoenix_type_files(phoenix: str) -> None:
    target: str = paths.concat([PHOENIX, phoenix])
    if not os.path.exists(target):
        make_phoenix_type_directory(phoenix)
    for number in NUMBERS:
        path: str = "{}/{}_{}.fits".format(target, phoenix, number)
        if not os.path.exists(path):
            with open(path, "w") as file:
                continue

def make_phoenix_catalog() -> None:
    if not os.path.exists(PHOENIX):
        make_phoenix_root_directory()
    path: str = "{}/catalog.fits".format(PHOENIX)
    if not os.path.exists(path):
        with open(path, "w") as file:
            pass

def make_phoenix_installed() -> None:
    if not os.path.exists(ASSETS):
        os.mkdir(ASSETS)
    make_phoenix_catalog()
    for phoenix in PHOENIXS:
        make_phoenix_type_files(phoenix)

def remove_phoenix() -> None:
    if os.path.exists(ASSETS):
        shutil.rmtree(ASSETS)

def test_is_phoenix_installed_when_fully_installed():
    # Arrange 
    remove_phoenix()
    os.mkdir(ASSETS)
    make_phoenix_installed()

    # Assert
    assert phoenix.is_phoenix_installed(ASSETS)

    # Clean Up
    remove_phoenix()

def test_is_phoenix_installed_when_fully_installed():
    # Arrange 
    remove_phoenix()
    os.mkdir(ASSETS)
    make_phoenix_catalog()

    # Assert
    assert not phoenix.is_phoenix_installed(ASSETS)

    # Clean Up
    remove_phoenix()
    
def test_is_phoenix_installed_when_not_installed():
    # Arrange
    remove_phoenix()

    # Assert
    assert not phoenix.is_phoenix_installed(ASSETS)

def test_make_phoenix_dirs_when_not_setup():
    # Arrange
    remove_phoenix()
    os.mkdir(ASSETS)

    # Act
    phoenix.make_phoenix_dirs(ASSETS)

    # Assert
    assert os.path.exists(paths.concat([ASSETS, "grid"]))
    assert os.path.exists(paths.concat([ASSETS, "grid/phoenix"]))
    assert os.path.exists(paths.concat([ASSETS, "grid/phoenix/phoenixm00"]))
    assert os.path.exists(paths.concat([ASSETS, "grid/phoenix/phoenixp03"]))
    
    # Clean Up
    remove_phoenix()

def test_make_phoenix_dirs_when_setup():
    # Arrange
    grid: str = paths.concat([ASSETS, "grid"])
    phoenixs: str = paths.concat([ASSETS, "grid", "phoenix"])
    phoenixm00: str = paths.concat([ASSETS, "grid", "phoenix", "phoenixm00"])
    phoenixp03: str = paths.concat([ASSETS, "grid", "phoenix", "phoenixp03"])

    remove_phoenix()
    os.mkdir(ASSETS)
    os.mkdir(grid)
    os.mkdir(phoenixs)
    os.mkdir(phoenixm00)
    os.mkdir(phoenixp03)

    # Act
    phoenix.make_phoenix_dirs(ASSETS)

    # Assert
    assert os.path.exists(grid)
    assert os.path.exists(phoenixs)
    assert os.path.exists(phoenixm00)
    assert os.path.exists(phoenixp03)
    
    # Clean Up
    remove_phoenix()
    
def test_make_phoenix_dirs_when_partially_setup():
    # Arrange
    remove_phoenix()
    os.mkdir(ASSETS)
    os.mkdir(paths.concat([ASSETS, "grid"]))

    # Act
    phoenix.make_phoenix_dirs(ASSETS)

    # Assert
    assert os.path.exists(paths.concat([ASSETS, "grid"]))
    assert os.path.exists(paths.concat([ASSETS, "grid/phoenix"]))
    assert os.path.exists(paths.concat([ASSETS, "grid/phoenix/phoenixm00"]))
    assert os.path.exists(paths.concat([ASSETS, "grid/phoenix/phoenixp03"]))
    
    # Clean Up
    remove_phoenix()

def test_install_phoenix_complete():
    # Arrange
    remove_phoenix()
    os.mkdir(ASSETS)

    # Act
    phoenix.install_phoenix(ASSETS, full = False)

    # Assert
    for file in PHOENIX_FILES:
        assert os.path.isfile(file)

    # Clean Up
    remove_phoenix()

def test_install_phoenix_when_partially_installed():
    # Arrange
    remove_phoenix()
    make_phoenix_installed()
    os.remove(PHOENIX_FILES[0])

    # Act
    phoenix.install_phoenix(ASSETS)

    # Assert 
    for file in PHOENIX_FILES:
        assert os.path.isfile(file)
    
def test_install_phoenix_when_fully_installed():
    # Arrange
    entry: str = "Hello world!"

    remove_phoenix()
    os.mkdir(ASSETS)
    os.mkdir(paths.concat([ASSETS, "grid"]))
    os.mkdir(paths.concat([ASSETS, "grid", "phoenix"]))
    os.mkdir(paths.concat([ASSETS, "grid", "phoenix", "phoenixm00"]))
    os.mkdir(paths.concat([ASSETS, "grid", "phoenix", "phoenixp03"]))

    for file in PHOENIX_FILES:
        with open(file, "w") as file:
            file.write(entry)
    
    # Act
    phoenix.install_phoenix(ASSETS)

    # Assert
    for file in PHOENIX_FILES:
        if not os.path.exists(file):
            raise ValueError

        with open(file, "r") as file:
            assert file.read() == entry


    # Clean Up
    remove_phoenix()

def test_set_phoenix_environ_when_not_set():
    # Arrange
    if os.environ.get("PYSYN_CDBS"):
        os.environ.pop("PYSYN_CDBS")

    # Act
    phoenix.set_phoenix_environ(ASSETS)

    # Assert
    assert os.environ["PYSYN_CDBS"] == ASSETS

def test_set_phoenix_environ_when_set():
    # Arrange
    if not os.environ.get("PYSYN_CDBS"):
        os.environ["PYSYN_CDBS"] = "ABCFU"

    # Act
    phoenix.set_phoenix_environ(ASSETS)

    # Assert
    assert os.environ["PYSYN_CDBS"] == ASSETS

def test_make_phoenix_spectra_when_root_valid():
    # Arrange
    spectra: float = phoenix.make_phoenix_spectra(".assets")

    print("PYSYN_CDBS", os.environ["PYSYN_CDBS"])

    # Assert
    assert spectra.shape[0] == 3

def test_make_phoenix_spectra_when_root_not_valid():
    # Arrange 
    remove_phoenix()
    
    # Act/Assert
    with pytest.raises(ValueError):
        spectra: float = phoenix.make_phoenix_spectra(ASSETS)

def test_clip_phoenix_spectra_in_range():
    # Arrange
    FILTER_MIN_WAVELENGTH: float = const.get_const_as_type("FILTER_MIN_WAVELENGTH", float)
    FILTER_MAX_WAVELENGTH: float = const.get_const_as_type("FILTER_MAX_WAVELENGTH", float)

    min_wavelength: float = FILTER_MIN_WAVELENGTH / 2.
    max_wavelength: float = FILTER_MAX_WAVELENGTH + FILTER_MIN_WAVELENGTH / 2.
    shape: int = 100

    spectra: float = np.array([
        np.linspace(min_wavelength, max_wavelength, shape),
        random.normal(random.PRNGKey(0), (shape,)),
        random.normal(random.PRNGKey(1), (shape,)),
    ], dtype = float)

    # Act
    out: float = phoenix.clip_phoenix_spectra(spectra)

    # Assert
    assert (out[0] >= FILTER_MIN_WAVELENGTH).all()
    assert (out[0] <= FILTER_MAX_WAVELENGTH).all()

def test_clip_phoenix_spectra_on_invalid_input():
    # Arrange
    FILTER_MIN_WAVELENGTH: float = const.get_const_as_type("FILTER_MIN_WAVELENGTH", float)

    min_wavelength: float = 0.0 
    max_wavelength: float = FILTER_MIN_WAVELENGTH
    shape: int = 100

    spectra: float = np.array([
        np.linspace(min_wavelength, max_wavelength, shape),
        random.normal(random.PRNGKey(0), (shape,)),
        random.normal(random.PRNGKey(1), (shape,)),
    ], dtype = float)

    # Act
    out: float = phoenix.clip_phoenix_spectra(spectra)

    print(out.shape)

    # Assert
    assert out.shape[1] == 0

