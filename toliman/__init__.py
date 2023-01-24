__author__ = "Jordan Dennis"

from .toliman import *


def _mkdir(path: str) -> None:
    import os
    import warnings

    try:
        os.mkdir(path)
    except FileExistsError as ferr:
        warnings.warn("Using existing {}.".format(path))

_mkdir("toliman/assets")


