import pytest
import os
import toliman.constants as const

PINEAPPLE: str = "PINEAPPLE"

def test_set_const():
    # Arrange
    const.set_const(PINEAPPLE, 1)

    # Assert
    assert os.environ[PINEAPPLE] == "1"

def test_set_const_changes_value():
    # Arrange
    if not os.environ.get(PINEAPPLE)== 1:
        os.environ[PINEAPPLE] = "1"

    # Act
    const.set_const(PINEAPPLE, 0)

    # Assert
    assert os.environ[PINEAPPLE] == "0"


