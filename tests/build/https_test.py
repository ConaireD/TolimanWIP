from toliman.build.https import (
    get_https_stream,
    dowload_byte_from_https,
    dowload_file_from_https,
)

from os import (mkdir, path, stat)
from shutil import (rmtree)

def test_get_https_stream_on_success():
    # Arrange
    stream: iter = get_https_stream("https://github.com")

    # Act/Assert
    assert stream.status_code == 200
    
def test_get_https_stream_on_failure():
    # Arrange/Act/Assert
    with pytest.raises(ValueError):
        stream: iter = get_https_stream("https://i'm/not/a/website.txt")

def test_download_byte_from_https_takes_byte():
    # Arrange
    mkdir("tmp")
    path: str = "tmp/jordan-dennis.html"
    url: str = "https://jordan-dennis.github.io"

    # Act
    download_byte_from_https(path, url)

    # Assert
    if not path.isfile(path):
        raise ValueError
    
    assert 0.8 < stat(path).st_size < 1.2

    # Clean Up
    rmtree("tmp")

def test_download_byte_from_https_skips_existing():
    # Arrange
    mkdir("tmp")
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
    rmtree("tmp")

def test_download_byte_from_https_on_failure():
    # Arrange
    mkdir("tmp")
    path: str = "tmp/jordan-dennis.html"
    url: str = "https://i/hate/everything/about/you/why/do/i/love/you"

    # Act/Assert
    with pytest.raises(ValueError):
        download_byte_from_https(path, url)

#def test_download_file_from_https_takes_file():
#def test_download_file_from_https_skips_existing():
#def test_download_file_from_https_on_failure():
