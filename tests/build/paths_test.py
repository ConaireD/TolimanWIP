from toliman.build.paths import (
    accumulate,
    concat
)

from pytest.mark import (
    parametrize
)

def test_accumulate_on_empty_entry():
    # Arrange
    entry: list = []

    # Act
    out: list = accumulate(entry)

    # Assert
    assert out == []

def test_accumulate_on_single_entry():
    # Arrange 
    entry: list = ["hi"]

    # Act 
    out: list = accumulate(entry)

    # Assert
    assert out == ["hi"]

def test_accumulate_on_multiple_entries(): # TODO: parametrise this.
    # Arrange
    entry: list = ["hi", "there", "how"]

    # Act
    out: list = accumulate(entry)

    # Assert 
    assert out == ["hi", "hi/there", "hi/there/how"]

def test_concat_on_empty_entry():
    # Arrange
    entry: list = []

    # Act
    out: list = concat(entry)

    # Assert
    assert out == ""

def test_concat_on_single_entry():

def test_concet_on_multiple_entries(): # TODO: parametrise this
