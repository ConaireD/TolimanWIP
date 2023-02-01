import pytest
import shutil
import os
import toliman.build.phoenix as phoenix
import toliman.build.paths as paths
import toliman.constants as const
import jax.numpy as np
import jax.random as random
import typing

class fixture(typing.Generic[typing.TypeVar("T")]): pass

ROOT: str = "tmp"

@pytest.mark.parametrize("full", [True])
@pytest.mark.parametrize("root", [ROOT])
def test_is_phoenix_installed_when_fully_installed(
        create_fake_phoenix_installation: callable,
    ) -> None:
    assert phoenix.is_phoenix_installed(ROOT)

@pytest.mark.parametrize("full", [False])
@pytest.mark.parametrize("root", [ROOT])
def test_is_phoenix_installed_when_partially_installed(
        create_fake_phoenix_installation: callable,
    ) -> None:
    assert not phoenix.is_phoenix_installed(ROOT)
    
@pytest.mark.parametrize("root", [ROOT])
def test_is_phoenix_installed_when_not_installed(
        remove_installation: callable,
    ) -> None:
    assert not phoenix.is_phoenix_installed(ROOT)

@pytest.mark.parametrize("root", [ROOT])
def test_make_phoenix_dirs_when_not_setup(
        remove_installation: fixture[None],
        list_phoenix_dirs: fixture[list]
    ) -> None:
    phoenix.make_phoenix_dirs(ROOT)
    assert all(os.path.isdir(pdir) for pdir in list_phoenix_dirs)

@pytest.mark.parametrize("full", [True])
@pytest.mark.parametrize("root", [ROOT])
def test_make_phoenix_dirs_when_setup(
        create_fake_phoenix_dirs: fixture[None],
    ) -> None:
    phoenix.make_phoenix_dirs(ROOT)

    # Assert
    assert os.path.isdir(grid)
    assert os.path.isdir(phoenixs)
    assert os.path.isdir(phoenixm00)
    assert os.path.isdir(phoenixp03)
    
# TODO:
def test_make_phoenix_dirs_when_partially_setup():
    # Arrange
    remove_installation()
    os.mkdir(ROOT)
    os.mkdir(paths.concat([ROOT, "grid"]))

    # Act
    phoenix.make_phoenix_dirs(ROOT)

    # Assert
    assert os.path.exists(paths.concat([ROOT, "grid"]))
    assert os.path.exists(paths.concat([ROOT, "grid/phoenix"]))
    assert os.path.exists(paths.concat([ROOT, "grid/phoenix/phoenixm00"]))
    assert os.path.exists(paths.concat([ROOT, "grid/phoenix/phoenixp03"]))
    
    # Clean Up
    remove_installation()

def test_install_phoenix_complete():
    # Arrange
    remove_installation()
    os.mkdir(ROOT)

    # Act
    phoenix.install_phoenix(ROOT, full = False)

    # Assert
    for file in PHOENIX_FILES:
        assert os.path.isfile(file)

    # Clean Up
    remove_installation()

def test_install_phoenix_when_partially_installed():
    # Arrange
    remove_installation()
    create_fake_phoenix_installation()
    os.remove(PHOENIX_FILES[0])

    # Act
    phoenix.install_phoenix(ROOT)

    # Assert 
    for file in PHOENIX_FILES:
        assert os.path.isfile(file)
    
def test_install_phoenix_when_fully_installed():
    # Arrange
    create_fake_phoenix_installation()
    
    # Act
    phoenix.install_phoenix(ROOT)

    # Assert
    for file in PHOENIX_FILES:
        if not os.path.exists(file):
            raise ValueError

        with open(file, "r") as file:
            assert not file.read()

    # Clean Up
    remove_installation()

def test_set_phoenix_environ_when_not_set():
    # Arrange
    if os.environ.get("PYSYN_CDBS"):
        os.environ.pop("PYSYN_CDBS")

    # Act
    phoenix.set_phoenix_environ(ROOT)

    # Assert
    assert os.environ["PYSYN_CDBS"] == ROOT

def test_set_phoenix_environ_when_set():
    # Arrange
    if not os.environ.get("PYSYN_CDBS"):
        os.environ["PYSYN_CDBS"] = "ABCFU"

    # Act
    phoenix.set_phoenix_environ(ROOT)

    # Assert
    assert os.environ["PYSYN_CDBS"] == ROOT

@pytest.mark.skipif(
    not phoenix.is_phoenix_installed(ROOT), 
    reason="No valid installation."
)
def test_make_phoenix_spectra_when_root_valid():
    # Arrange
    spectra: float = phoenix.make_phoenix_spectra(".assets")

    # Assert
    assert spectra.shape[0] == 3

def test_make_phoenix_spectra_when_root_not_valid():
    # Arrange 
    remove_installation()
    
    # Act/Assert
    with pytest.raises(ValueError):
        spectra: float = phoenix.make_phoenix_spectra(ROOT)

def test_clip_phoenix_spectra_in_range():
    # Arrange
    min_wavelength: float = FILTER_MIN_WAVELENGTH / 2.
    max_wavelength: float = FILTER_MAX_WAVELENGTH + FILTER_MIN_WAVELENGTH / 2.

    spectra: float = get_spectra(min_wavelength, max_wavelength)

    # Act
    out: float = phoenix.clip_phoenix_spectra(spectra)

    # Assert
    assert (out[0] >= FILTER_MIN_WAVELENGTH).all()
    assert (out[0] <= FILTER_MAX_WAVELENGTH).all()

def test_clip_phoenix_spectra_on_invalid_input():
    # Arrange
    spectra: float = get_spectra(0.0, FILTER_MIN_WAVELENGTH) 

    # Act
    out: float = phoenix.clip_phoenix_spectra(spectra)

    # Assert
    assert out.shape[1] == 0

# TODO: The way the clipping happens will be a problem for smaller arrays
# TODO: Implement a get_spectra fixture
def test_resample_phoenix_spectra_produces_correct_shape():
    spectra: float = get_spectra(FILTER_MIN_WAVELENGTH, FILTER_MAX_WAVELENGTH)
    out_shape: int = 10

    # Act
    out: float = phoenix.resample_phoenix_spectra(spectra, out_shape)

    # Assert
    assert out.shape == (3, out_shape)
    
def test_save_phoenix_spectra_makes_file():
    # Arrange
    spectra: float = get_spectra(FILTER_MIN_WAVELENGTH, FILTER_MAX_WAVELENGTH)

    if os.path.isdir(ROOT): 
        shutil.rmtree(ROOT)
     
    os.mkdir(ROOT)

    # Act
    phoenix.save_phoenix_spectra("tmp", spectra)

    # Assert
    assert os.path.isfile("tmp/spectra.csv")

def test_save_phoenix_spectra_has_headings():
    # Arrange
    spectra: float = get_spectra(FILTER_MIN_WAVELENGTH, FILTER_MAX_WAVELENGTH)

    if os.path.isdir(ROOT): 
        shutil.rmtree(ROOT)
     
    os.mkdir(ROOT)

    # Act
    phoenix.save_phoenix_spectra("tmp", spectra)

    # Assert
    with open("tmp/spectra.csv", "r") as file:
        header: str = next(file)
        titles: list = header.strip().split(",")

        assert titles[0].strip() == "alpha cen a waves (m)"
        assert titles[1].strip() == "alpha cen a flux (W/m/m)"
        assert titles[2].strip() == "alpha cen b flux (W/m/m)"

def test_save_phoenix_spectra_has_correct_lines():
    # Arrange
    spectra: float = get_spectra(FILTER_MIN_WAVELENGTH, FILTER_MAX_WAVELENGTH)

    if os.path.isdir(ROOT): 
        shutil.rmtree(ROOT)
     
    os.mkdir(ROOT)

    # Act
    phoenix.save_phoenix_spectra("tmp", spectra)

    # Assert
    with open("tmp/spectra.csv", "r") as file:
        num_lines: int = len(file.readlines())
        assert num_lines == 101
