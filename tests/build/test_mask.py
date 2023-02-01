import pytest
import toliman.build.paths as paths
import toliman.build.mask as mask
import os
import shutil
import jax.numpy as np

ROOT: str = "tmp"

def test_is_mask_installed_when_installed() -> None:
    # Arrange
    if os.path.isdir(ROOT):
        shutil.rmtree(ROOT)

    os.mkdir(ROOT)
    open(paths.concat([ROOT, "mask.npy"]), "w").close()

    # Act
    assert mask.is_mask_installed(ROOT)

    # Clean Up
    if os.path.isdir(ROOT):
        shutil.rmtree(ROOT)

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
    # Arrange
    if os.path.isdir(ROOT):
        shutil.rmtree(ROOT)

    # Act
    mask.install_mask(ROOT)

    # Assert
    masks: float = np.load(paths.concat([ROOT, "mask.npy"]))
    assert masks.shape == (1024, 1024)

    # Clean Up
    if os.path.isdir(ROOT):
        shutil.rmtree(ROOT)
