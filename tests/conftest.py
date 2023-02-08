import pytest
import os
import shutil

@pytest.fixture
def remove_installation(root: str) -> None:
    rm: callable = lambda root: if os.path.isdir(root): shutil.rmtree(root)
    rm(root)
    yield
    rm(root)

