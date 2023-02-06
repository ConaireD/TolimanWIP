import pytest
import toliman.build.paths as paths
import toliman.build.mask as mask
import os
import shutil
import jax.numpy as np
import typing

class fixture(typing.Generic[typing.TypeVar("T")]): pass

ROOT: str = "tmp"

@pytest.mark.parametrize("root", [ROOT])
def test_is_mask_installed_when_installed(
        root: str,
        remove_installation: fixture[None],
    ) -> None:
    """
    Can mask.is_mask_installed detect and existing installation?

    Fixtures
    --------
    remove_installation: fixture[None],
        Ensures there is no installation before and after the test.

    Parameters
    ----------
    root: str = ROOT
        Where to look for the installation. Directly parametrises 
        mask.is_mask_installed. Indirectly parametrizes remove_installation.
    """
    open(paths.concat([root, "mask.npy"]), "w").close()
    assert mask.is_mask_installed(root)

def test_is_mask_installed_when_not_installed() -> None:
    # Arrange
    if os.path.isdir(ROOT):
        shutil.rmtree(ROOT)

    os.mkdir(ROOT)

    # Act 
    assert not mask.is_mask_installed(ROOT)

    # Clean Up
    if os.path.isdir(ROOT):
        shutil.rmtree(ROOT)

def test_install_mask_creates_file() -> None:
    # Arrange
    if os.path.isdir(ROOT):
        shutil.rmtree(ROOT)

    # Act
    mask.install_mask(ROOT)

    # Assert
    assert os.path.isfile(paths.concat([ROOT, "mask.npy"]))

    # Clean Up
    if os.path.isdir(ROOT):
        shutil.rmtree(ROOT)

def test_install_mask_has_correct_shape() -> None:
    mask.install_mask(ROOT)
    masks: float = np.load(paths.concat([ROOT, "mask.npy"]))
    assert masks.shape == (1024, 1024)
