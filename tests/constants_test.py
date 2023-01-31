import pytest
import os
import toliman.constants as const

PINEAPPLE: str = "PINEAPPLE"

def test_set_const() -> None:
    # Arrange
    const.set_const(PINEAPPLE, 1)

    # Assert
    assert os.environ[PINEAPPLE] == "1"

    # Clean Up
    if os.environ.get(PINEAPPLE):
        os.environ.pop(PINEAPPLE)

def test_set_const_changes_value() -> None:
    # Arrange
    if os.environ.get(PINEAPPLE):
        os.environ[PINEAPPLE] = "1"

    # Act
    const.set_const(PINEAPPLE, 0)

    # Assert
    assert os.environ[PINEAPPLE] == "0"

    # Clean Up
    if os.environ.get(PINEAPPLE):
        os.environ.pop(PINEAPPLE)


@pytest.mark.parametrize("types", [float, int, str])
def test_get_const_as_type_has_correct_type(types: type) -> None:
    # Arrange
    os.environ[PINEAPPLE] = "1"

    # Act
    pineapple: object = const.get_const_as_type(PINEAPPLE, types)

    # Assert
    assert isinstance(pineapple, types)

    # Clean Up
    if os.environ.get(PINEAPPLE):
        os.environ.pop(PINEAPPLE)

def test_get_const_as_type_fails() -> None:
    # Arrange
    if os.environ.get(PINEAPPLE):
        os.environ.pop(PINEAPPLE)

    # Act/Assert
    with pytest.raises(KeyError):
        const.get_const_as_type(PINEAPPLE, str)

    
