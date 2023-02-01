import os
import warnings
import toliman.build.mask as mask
import toliman.build.phoenix as phoenix
import toliman.build.background as bg
import toliman.constants as const

__author__ = "Jordan Dennis"
__all__ = [
    "is_toliman_installed",
    "install_toliman", # TODO: force
]

TOLIMAN_HOME: str = const.get_const_as_type("TOLIMAN_HOME", str) 

def is_toliman_installed(root: str = TOLIMAN_HOME) -> bool:
    """
    Checks if all the assets files for toliman are installed.

    Parameters
    ----------
    root: str = TOLIMAN_HOME
        The directory to search in. This should be TOLIMAN_HOME 
        but I have enabled other options if multiple installations 
        exist.

    Returns
    -------
    installed: bool
        True if all the components are installed, else False.

    Examples
    --------
    >>> import toliman.build as build
    >>> import toliman.build.mask as mask
    >>> import toliman.build.phoenix as phoenix
    >>> import toliman.build.background as bg
    >>> import shutil 
    >>> import os
    >>> if os.path.isdir("tmp"):
    ...     shutil.rmtree("tmp")
    >>> os.mkdir("tmp")
    >>> mask.install_mask("tmp")
    >>> mask.is_mask_installed("tmp")
    ::: True
    >>> phoenix.is_phoenix_installed("tmp")
    ::: False
    >>> bg.is_background_installed("tmp")
    ::: False
    >>> build.is_toliman_installed("tmp")
    ::: False
    >>> phoenix.install_phoenix("tmp")
    >>> bg.install_background_stars("tmp")
    >>> build.is_toliman_installed("tmp")
    ::: True
    """
    component_installations: list = [
        phoenix.is_phoenix_installed(),
        mask.is_mask_installed(),
        bg.are_background_stars_installed(),
    ]

    return all(component_installations)

def build(root: str = TOLIMAN_HOME):
    print("Building `toliman`!")
    print("-------------------")

    if not _is_phoenix_installed():
        print("Installing phoenix...")
        _install_phoenix()
        print("Done!")

    print("Saving spectral model...")
    _simulate_alpha_cen_spectra()
    print("Done!")

    print("Saving background stars...")
    _simulate_background_stars()
    print("Done!")

    if not _is_mask_installed():
        print("Installing mask...")
        _install_mask()
        print("Done!")

    print("`toliman` built!")
