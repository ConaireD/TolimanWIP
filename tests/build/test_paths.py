import toliman.build.paths as paths 
import os 
import shutil

def test_accumulate_on_empty_entry():
    # Arrange
    entry: list = []

    # Act
    out: list = paths.accumulate(entry)

    # Assert
    assert out == []

def test_accumulate_on_single_entry():
    # Arrange 
    entry: list = ["hi"]

    # Act 
    out: list = paths.accumulate(entry)

    # Assert
    assert out == ["hi"]

def test_accumulate_on_multiple_entries(): # TODO: parametrise this.
    # Arrange
    entry: list = ["hi", "there", "how"]

    # Act
    out: list = paths.accumulate(entry)

    # Assert 
    assert out == ["hi", "hi/there", "hi/there/how"]

def test_concat_on_empty_entry():
    # Arrange
    entry: list = []

    # Act
    out: str = paths.concat(entry)

    # Assert
    assert out == ""

def test_concat_on_single_entry():
    # Arrange 
    entry: list = ["hi"]

    # Act
    out: str = paths.concat(entry)

    #Assert
    assert out == "hi"

def test_concet_on_multiple_entries(): # TODO: parametrise this
    # Arrange
    entry: list = ["hi", "there", "how"]

    # Act
    out: str = paths.concat(entry)

    # Assert
    assert out == "hi/there/how"

def test_mkdir_and_parents_on_single_dir() -> None:    
    # Arrange
    entry: str = "tmp"
    if os.path.exists(entry):
        shutil.rmtree(entry)

    # Act
    paths.mkdir_and_parents(entry)

    # Assert
    assert os.path.isdir(entry)

    # Clean Up
    if os.path.exists(entry):
        shutil.rmtree(entry)

def test_mkdir_and_parents_on_many_dirs() -> None:    
    # Arrange
    root: str = "tmp"
    if os.path.exists(root):
        shutil.rmtree(root)

    entry: str = "{}/hi/there/how/are/you".format(root)

    # Act
    paths.mkdir_and_parents(entry)

    # Assert
    assert os.path.isdir(entry)

    # Clean Up
    if os.path.exists(root):
        shutil.rmtree(root)

