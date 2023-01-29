from toliman.build.paths import (
    accumulate,
    concat
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
    out: str = concat(entry)

    # Assert
    assert out == ""

def test_concat_on_single_entry():
    # Arrange 
    entry: list = ["hi"]

    # Act
    out: str = concat(entry)

    #Assert
    assert out == "hi"

def test_concet_on_multiple_entries(): # TODO: parametrise this
    # Arrange
    entry: list = ["hi", "there", "how"]

    # Act
    out: str = concat(entry)

    # Assert
    assert out == "hi/there/how"
