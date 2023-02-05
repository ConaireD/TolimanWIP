import requests
import os
import warnings

__author__ = "Jordan Dennis"
__all__ = [
    "get_https_stream",
    "download_file_from_https",
    "download_byte_from_https",
]

BYTE: int = 1024

def get_https_stream(url: str) -> object:
    """
    Load a website as an iterable.

    This function is used to lazily load a website. Although the total 
    runtime may be slower it allows for much faster tests by permitting
    only a small chunk of the website to be downloaded at a time.
    `get_https_stream` raises a `ValueError` upon a failure.

    Parameters
    ----------
    url: str 
        The website to visit.

    Returns
    -------
    stream: object 
        An iterable representing the website data.

    Examples
    --------
    >>> get_https_stream("https:/im/not/real.com")
    ::: ValueError
    """
    response: iter = requests.get(url, stream=True)
    
    if not response.status_code == 200:
        raise ValueError

    return response

def download_file_from_https(path: str, url: object) -> None:
    """
    Download a file from the internet.

    This will save a file on the internet to `path`. It is assumed 
    that `path` does not currently exist. If it does a warning is 
    given and the download is skipped.

    Parameters
    ----------
    path: str
        The location to save the file to.
    url: str
        The address of the information on the internet.

    Examples
    --------
    >>> import os
    >>> os.mkdir("tmp")
    >>> with open("tmp/text.txt", "w") as file:
    >>>     pass
    >>> download_file_from_https("tmp/text.txt", "github.com")
    ::: "`path` exists. Skipping download"
    >>> download_file_from_https("tmp/github.txt", "github.com")
    >>> os.path.isfile("tmp/github.txt")
    ::: True
    """
    if os.path.isfile(path):
        warnings.warn("`path` exists. Skipping download.")
        return None

    response: iter = get_https_stream(url).iter_content(BYTE)

    with open(path, "wb") as file:
        for data in response:
            file.write(data)

def download_byte_from_https(path: str, url: object) -> None:
    """
    Download a single byte of a file from the internet.

    This will save the first byte of a file on the internet to `path`.
    If `path` already points to a file a warning is printed and the 
    download is skipped.

    Parameters
    ----------
    path: str
        The location to save the file to.
    url: str
        The address of the information on the internet.

    Examples
    --------
    >>> import os
    >>> os.mkdir("tmp")
    >>> with open("tmp/text.txt", "w") as file:
    >>>     pass
    >>> download_byte_from_https("tmp/text.txt", "github.com")
    ::: "`path` exists. Skipping download"
    >>> download_byte_from_https("tmp/github.txt", "github.com")
    >>> os.path.isfile("tmp/github.txt")
    ::: True
    """
    if os.path.isfile(path):
        warnings.warn("`path` exists. Skipping download.")
        return None

    response: iter = get_https_stream(url).iter_content(BYTE)

    with open(path, "wb") as file:
        file.write(next(response))
