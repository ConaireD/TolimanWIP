from toliman.build.https import (
    get_https_stream,
    download_byte_from_https,
    download_file_from_https,
)

from pytest import (raises)
import os
import shutil

def test_get_https_stream_on_success():
    # Arrange
    stream: iter = get_https_stream("https://github.com")

    # Act/Assert
    assert stream.status_code == 200
    
def test_download_byte_from_https_takes_byte():
    # Arrange
    if os.path.exists("tmp"):
        shutil.rmtree("tmp")

    os.mkdir("tmp")
    path: str = "tmp/jordan-dennis.html"
    url: str = "https://jordan-dennis.github.io"

    # Act
    download_byte_from_https(path, url)

    # Assert
    if not os.path.isfile(path):
        raise ValueError
    
    assert 0 < os.stat(path).st_size < 2000

    # Clean Up
    shutil.rmtree("tmp")

def test_download_byte_from_https_skips_existing():
    # Arrange
    if os.path.exists("tmp"):
        shutil.rmtree("tmp")

    os.mkdir("tmp")
    path: str = "tmp/jordan-dennis.html"
    url: str = "https://jordan-dennis.github.io"
    entry: str = "Hello world!"

    with open(path, "w") as file:
        file.write(entry)

    # Act
    download_byte_from_https(path, url)

    # Assert
    with open(path, "r") as file:
        assert entry == file.read()

    # Clean Up
    shutil.rmtree("tmp")

def test_download_file_from_https_takes_file():
    # Arrange
    if os.path.exists("tmp"):
        shutil.rmtree("tmp")

    os.mkdir("tmp")
    path: str = "tmp/jordan-dennis.html"
    url: str = "https://jordan-dennis.github.io"

    # Act
    download_file_from_https(path, url)

    # Assert
    if not os.path.isfile(path):
        raise ValueError

    with open(path, "r") as file:
        source: str = file.read()
        assert source.startswith("<!DOCTYPE html>")
        assert source.endswith("</body>")

    # Clean Up
    shutil.rmtree("tmp")

def test_download_file_from_https_skips_existing():
    # Arrange
    if os.path.exists("tmp"):
        shutil.rmtree("tmp")

    os.mkdir("tmp")
    path: str = "tmp/jordan-dennis.html"
    url: str = "https://jordan-dennis.github.io"
    entry: str = "Hello world!"

    with open(path, "w") as file:
        file.write(entry)

    # Act
    download_file_from_https(path, url)

    # Assert
    with open(path, "r") as file:
        assert entry == file.read()

    # Clean Up
    shutil.rmtree("tmp")
