import os
import toliman.build.paths as paths
import toliman.build.https as https
import toliman.constants as const

__author__ = "Jordan Dennis"
__all__ = [
    "is_mask_installed",
    "install_mask",
]

MASK_HOME: str = "https://github.com/ConaireD/TolimanWIP/raw/mask/mask.npy"
TOLIMAN_HOME: str = const.get_const_as_type("TOLIMAN_HOME", str)
MASK_FILE: str = "mask.npy"

def is_mask_installed(root: str) -> bool:
    """
    Check if the mask is installed.

    Parameters
    ----------
    root: str
        The directory to search for an installation in.

    Returns
    -------
    installed: bool
        True is the mask is installed else false.

    Examples
    --------
    >>> import os
    >>> os.mkdir("tmp")
    >>> is_mask_installed("tmp")
    ::: False
    >>> open("tmp/mask.npy", "w").close()
    >>> is_mask_installed("tmp")
    ::: True
    """
    if not os.path.isdir(root):
        return False
        
    return os.path.isfile(paths.concat([root, MASK_FILE]))

def install_mask(root: str) -> bool:
    """
    Install phoenix from the web.

    Parameters
    ----------
    root: str
        The directory to search for an installation in.

    Examples
    --------
    >>> import os
    >>> install_mask(".assets")
    >>> os.path.isfile(".assets/mask.npy")
    ::: True
    """
    paths.mkdir_and_parents(root)
    path: str = paths.concat([root, MASK_FILE])
    https.download_file_from_https(path, MASK_HOME)
