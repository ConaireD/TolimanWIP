"""md
## Overview
Unfortunately, the toliman forwards model is not completely self contained,
since the datafiles need to be shared and it is inefficient to do so using 
`git`. We automated this process so that it was easy for the user, writing 
code to download the files from the interent. Unfortunately some of the 
files are very large so it takes a long time to download them (slower or 
faster depending on your connection). 

A focus of the early forwards model development was ensuring that it was 
rigorously tested so that later users could more easily extend and debug. 
However, unit tests have to be faster and completely downloading all of
the dependencies meant that the tests would take hours to run, rendering 
them useless. As a result, for testing purposes this submodule also 
provides functionality to download a single byte instead of an entire 
file. This is mostly used for testing.

!!! note
    Good tests are repeatable i.e. deterministic. While we speant some time
    making sure that the tests were fast, the internet is the interent and 
    connections can fail. This means that the tests are not completely 
    deterministic, but, if your connection is good, then they are effectively
    deterministic.

## API
??? note "`get_https_stream`"
    ::: toliman.build.https.get_https_stream

??? note "`download_file_from_https`"
    ::: toliman.build.https.download_file_from_https

??? note "`download_byte_from_https`"
    ::: toliman.build.https.download_byte_from_https

"""

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
