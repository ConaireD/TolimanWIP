import pytest
import toliman.build.paths as paths
import toliman.build.mask as mask
import os
import shutil

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

    os.mdkir(ROOT)

    # Act 
    assert not mask.is_mask_installed(ROOT)

    # Clean Up
    if os.path.isdir(ROOT):
        shutil.rmtree(ROOT)
