"""md
## API
??? note `is_toliman_installed`
    ::: toliman.build.build.is_toliman_installed

??? note `install_toliman", # TODO: force`
    ::: toliman.build.build.install_toliman", # TODO: force

"""

import os
import warnings
import termcolor
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

def color_str_as_code(string: str) -> None:
    """
    Format markdown style code blocks.

    Parameters
    ----------
    string: str
        The string to format with markdown code blocks.

    Returns
    -------
    string: str
        The same string with ANSI escape charaters around the 
        code blocks.
    """
    start: int = string.find("`")
    if start > 0:
        end: int = string.find("`", start)
        if end < 0:
            raise ValueError("Code segement was not terminated.")
        code: str = string[start:end]
        c_code: str = termcolor.colored(code, "light_grey", "on_dark_grey")
        return string[:start - 1] + c_code + string[end:]
    return string 

def install_toliman(
        number_of_wavelengths: int, 
        root: str = TOLIMAN_HOME, /, 
        force: bool = False
    ) -> None:
    """
    Install all the resource files for toliman.

    Parameters
    ----------
    number_of_wavelengths: int
        How many wavelengths should be saved in the spectra 
        generated using `phoenix` models?
    root: str = TOLIMAN_HOME
        The directory to search in. This should be TOLIMAN_HOME 
        but I have enabled other options if multiple installations 
        exist.
    force: bool = False
        If True then any existing installation will be deleted. 
        Otherwise existsing files will be skipped if a partial 
        implementation exists.

    Examples
    --------
    >>> import toliman.build as build
    >>> build.get_toliman_home()
    ::: .assets
    >>> build.is_toliman_installed()
    ::: False
    >>> build.install_toliman()
    >>> build.is_toliman_installed()
    ::: True
    """
    print(color_str_as_code("Building `toliman`!"))

    if not phoenix.is_phoenix_installed(root) or force:
        print("Installing phoenix...")
        phoenix.install_phoenix(root, full = True)
        print("Done!")

    if not phoenix.is_spectra_installed(root) or force: 
        print("Installing spectra...")
        phoenix.install_spectra(root, number_of_wavelengths)
        print("Done!")

    if not bg.are_background_stars_installed(root) or force:
        print("Installing background stars...")
        bg.install_background_stars(root)
        print("Done!")

    if not mask.is_mask_installed(root) or force:
        print("Installing mask...")
        mask.install_mask(root)
        print("Done!")

    print(color_str_as_code("`toliman` built!"))
