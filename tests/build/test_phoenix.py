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
FILTER_MIN: float = const.get_const_as_type("FILTER_MIN_WAVELENGTH", float)
FILTER_MAX: float = const.get_const_as_type("FILTER_MAX_WAVELENGTH", float)

@pytest.mark.parametrize("full", [True])
@pytest.mark.parametrize("root", [ROOT])
def test_is_phoenix_installed_when_fully_installed(
        create_fake_phoenix_installation: None,
    ) -> None:
    """
    phoenix.is_phoenix_installed identifies an existing installation.

    Fixtures
    --------
    create_fake_phoenix_installation: fixture[None]
        Cleans the testing directory and produces a "fake" installation
        of phoenix. "fake" is used to describe a complete installation 
        with empty data files. This fixture automatically handles the 
        setup and teardown process via the fixture remove_installation.

    Parameters
    ----------
    full: True
        Indirect parametrisation of create_fake_phoenix_installation.
        Ensures that all mock files are created.
    root: ROOT
        Indirect parametrisation of create_fake_phoenix_installation.
        Identifies where to create the mock installation. Also used 
        directly to tesll phoenix.is_phoenix_installed where to look.
    """
    assert phoenix.is_phoenix_installed(ROOT)

@pytest.mark.parametrize("full", [False])
@pytest.mark.parametrize("root", [ROOT])
def test_is_phoenix_installed_when_partially_installed(
        create_fake_phoenix_installation: callable,
    ) -> None:
    """
    phoenix.is_phoenix_installed identifies an incomplete installation.

    Fixtures
    --------
    create_fake_phoenix_installation: fixture[None]
        Cleans the testing directory and produces a "fake" installation
        of phoenix. "fake" is used to describe a complete installation 
        with empty data files. This fixture automatically handles the 
        setup and teardown process via the fixture remove_installation.

    Parameters
    ----------
    full: True
        Indirect parametrisation of create_fake_phoenix_installation.
        Ensures that all mock files are created.
    root: ROOT
        Indirect parametrisation of create_fake_phoenix_installation.
        Identifies where to create the mock installation. Also used 
        directly to tesll phoenix.is_phoenix_installed where to look.
    """
    assert not phoenix.is_phoenix_installed(ROOT)
    
@pytest.mark.parametrize("root", [ROOT])
def test_is_phoenix_installed_when_not_installed(
        remove_installation: callable,
    ) -> None:
    """
    phoenix.is_phoenix_installed identifies no installation.

    Fixtures
    --------
    remove_installation: fixture[None]
        Ensures that there is no installation of phoenix, before and
        after the test.

    Parameters
    ----------
    root: str = ROOT
        Indirect parametrisation of remove_installation.
        Direct parametrisation of phoenix.is_phoenix_installed,
        identifying where to look.
    """
    assert not phoenix.is_phoenix_installed(ROOT)

@pytest.mark.parametrize("root", [ROOT])
def test_make_phoenix_dirs_when_not_setup(
        remove_installation: fixture[None],
        list_phoenix_dirs: fixture[list]
    ) -> None:
    """
    phoenix.make_phoenix_dirs creates the appropriate directories.

    Fixtures
    --------
    remove_installation: fixture[None]
        Ensures that there is no installation of phoenix, before and
        after the test.
    list_phoenix_dirs: fixture[list]
        The directories storing the phoenix data files.

    Parameters
    ----------
    root: str = ROOT
        Indirect parametrisation of remove_installation identifying 
        target. Indirect parametrisation of list_phoenix_dirs 
        identifying target. Direct parametrisation of phoenix.make_phoenix_dirs
        identifying target.
    """
    phoenix.make_phoenix_dirs(ROOT)
    assert all(os.path.isdir(pdir) for pdir in list_phoenix_dirs)

@pytest.mark.parametrize("full", [True])
@pytest.mark.parametrize("root", [ROOT])
def test_make_phoenix_dirs_when_setup(
        create_fake_phoenix_dirs: fixture[None],
        list_phoenix_dirs: fixture[list],
    ) -> None:
    """
    phoenix.make_phoenix_dirs does not alter the existing installation.

    Fixtures
    --------
    create_fake_phoenix_dirs: fixture[None]
        Makes the tree structure for the phoenix directories. Automatically
        handles the setup and teardown via the remove_installation 
        fixture.
    list_phoenix_dirs: fixture[list]
        The directories storing the phoenix data files.

    Parameters
    ----------
    root: str = ROOT
        Indirect parametrisation of create_fake_phoenix_dirs identifying 
        target. Direct parametrisation of phoenix.make_phoenix_dirs
        identifying target.
    full: bool = True
        Indirect parametrisation of create_fake_phoenix_dirs 
        specifying the creation of the complete tree.
    """
    phoenix.make_phoenix_dirs(ROOT)
    assert all(os.path.isdir(pdir) for pdir in list_phoenix_dirs)

@pytest.mark.parametrize("full", [False])
@pytest.mark.parametrize("root", [ROOT])
def test_make_phoenix_dirs_when_partially_setup(
        create_fake_phoenix_dirs: fixture[None],
        list_phoenix_dirs: fixture[list],
    ) -> None:
    """
    phoenix.make_phoenix_dirs finishes the existing installation.

    Fixtures
    --------
    create_fake_phoenix_dirs: fixture[None]
        Makes the tree structure for the phoenix directories. Automatically
        handles the setup and teardown via the remove_installation 
        fixture.
    list_phoenix_dirs: fixture[list]
        The directories storing the phoenix data files.

    Parameters
    ----------
    root: str = ROOT
        Indirect parametrisation of create_fake_phoenix_dirs identifying 
        target. Direct parametrisation of phoenix.make_phoenix_dirs
        identifying target.
    full: bool = False
        Indirect parametrisation of create_fake_phoenix_dirs 
        specifying the creation of an incomplete tree.
    """
    phoenix.make_phoenix_dirs(ROOT)
    assert all(os.path.isdir(pdir) for pdir in list_phoenix_dirs)

@pytest.mark.parametrize("root", [ROOT])
def test_install_phoenix_when_not_installed(
        root: str,
        remove_installation: fixture[None],
        list_phoenix_files: fixture[list],
    ) -> None:
    """
    Does `phoenix.install_phoenix` install phoenix?

    Fixtures
    --------
    remove_installation: fixture[None]
        Ensures that there is no installation of phoenix, before and
        after the test.
    list_phoenix_files: fixture[list]
        The files storing the phoenix data.

    Parameters
    ----------
    root: str = ROOT
        Indirect parametrization of remove_installation and list_phoenix_files.
        Directly used by install_phoenix.
    """
    phoenix.install_phoenix(root, full = False)
    for file in list_phoenix_files: assert os.path.isfile(file)

@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("root", [ROOT])
def test_install_phoenix_when_partially_installed(
        root: str,
        create_fake_phoenix_installation: fixture[None],
        list_phoenix_files: fixture[list],
    ) -> None:
    """
    Does phoenix.install_phoenix finish an incomplete installation?
    Does phoenix.install_phoenix ignore a complete installation?

    Fixtures
    --------
    create_fake_phoenix_installation: fixture[None]
        Generates the phoenix files, but does not instantiate any values. 
        This fixture was created for speed.
    list_phoenix_files: fixture[list]
        Produces a list of all the phoenix files.

    Parameters
    ----------
    root: str = ROOT
        The directory to install phoenix in. Direct parametrisation of 
        phoenix.install_phoenix and indirect parametrization of 
        create_fake_phoenix_installation and list_phoenix_files.
    full: bool = False
        Install the first byte only. Indirect parametrisation of 
        create_fake_phoenix_installation, do not instantiate all 
        files.
    """
    phoenix.install_phoenix(root, full = False)
    for file in list_phoenix_files: assert os.path.isfile(file)

def test_set_phoenix_environ_when_not_set(
        setup_and_teardown_phoenix_environ: fixture[None],
    ) -> None:
    """
    Does phoenix.set_phoenix_environ set PYSYN_CDBS?

    Fixtures
    --------
    setup_and_teardown_phoenix_environ: fixture[None]
        Ensure that PYSYN_CDBS is not defined before and after the test.
    """
    phoenix.set_phoenix_environ(ROOT)
    assert os.environ["PYSYN_CDBS"] == ROOT

def test_set_phoenix_environ_when_set(
        setup_and_teardown_phoenix_environ: fixture[None],
    ) -> None:
    """
    Does phoenix.set_phoenix_environ set PYSYN_CDBS when PYSYN_CDBS is 
    already set?

    Fixtures
    --------
    setup_and_teardown_phoenix_environ: fixture[None]
        Ensure that PYSYN_CDBS is not defined before and after the test.
    """
    os.environ["PYSYN_CDBS"] = "ABCFU"
    phoenix.set_phoenix_environ(ROOT)
    assert os.environ["PYSYN_CDBS"] == ROOT

@pytest.mark.skipif(
    not phoenix.is_phoenix_installed(".assets"), 
    reason="No valid installation."
)
def test_make_phoenix_spectra_when_root_valid() -> None:
    """
    When phoenix is installed, does make_phoenix_spectra create spectra?
    """
    spectra: float = phoenix.make_phoenix_spectra(".assets")
    assert spectra.shape[0] == 3

@pytest.mark.parametrize("root", [ROOT])
def test_make_phoenix_spectra_when_root_not_valid(
        root: str,
        remove_installation: fixture[None],
    ) -> None:
    """
    When phoenix is not installed, does make_phoenix_spectra raise a 
    ValueError?

    Fixtures
    --------
    remove_installation: fixture[None],
        Ensures that there is not installation in the root directory before 
        and after the test.

    Parameters
    ----------
    root: str = ROOT
        Points where to look for phoenix. Direct parametrisation of 
        phoenix.set_phoenix_environ and phoenix.make_phoenix_spectra.
        Indirect parametrisation of remove_installation.
    """
    phoenix.set_phoenix_environ(root)
    with pytest.raises(ValueError):
        spectra: float = phoenix.make_phoenix_spectra(ROOT)

@pytest.mark.parametrize("min_", [FILTER_MIN / 2.])
@pytest.mark.parametrize("max_", [FILTER_MAX + FILTER_MIN / 2.])
def test_clip_phoenix_spectra_in_range(
        make_fake_spectra: fixture[None],
    ) -> None:
    """
    Does phoenix.clip_phoenix_spectra take within the filter?

    Fixtures
    --------
    make_fake_spectra: fixture[None],
        Generate a simplified and fake spectrum.
    
    Parameters
    ----------
    min_: float, meters
        The shortest wavelength. Indirectly parametrizes make_fake_spectra.
    max_: float, meters
        The longest wavelength. Indirectly parametrizes make_fake_spectra.
    """
    out: float = phoenix.clip_phoenix_spectra(make_fake_spectra)
    assert (out[0] >= FILTER_MIN).all()
    assert (out[0] <= FILTER_MAX).all()

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
