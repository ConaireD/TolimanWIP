import os
import toliman.build.paths as paths
import toliman.build.https as https
import toliman.math as math
import warnings
import jax.numpy as np

__author__ = "Jordan Dennis"
__all__ = [
    "is_phoenix_installed",
    "install_phoenix",
    "make_phoenix_dirs",
    "make_phoenix_spectra",
    "save_phoenix_spectra",
    "clip_phoenix_spectra",
    "resample_phoenix_spectra",
]

HOME: str = "grid/phoenix"
URL: str = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/phoenix"
M00: str = "phoenixm00"
P03: str = "phoenixp03"
NUMS: list = [5200, 5300, 5700, 5800]
WAVES: int = 0
ALPHA_CEN_A: int = 1
ALPHA_CEN_B: int = 2
PATHS: str = ["catalog.fits"] + [
    "{}/{}_{}.fits".format(pnx, pnx, num) 
        for num in NUMS 
        for pnx in [M00, P03]
]

ALPHA_CEN_A_SURFACE_TEMP: float = 5790.0
ALPHA_CEN_A_METALICITY: float = 0.2
ALPHA_CEN_A_SURFACE_GRAV: float = 4.0

ALPHA_CEN_B_SURFACE_TEMP: float = 5260.0
ALPHA_CEN_B_METALICITY: float = 0.23
ALPHA_CEN_B_SURFACE_GRAV: float = 4.37

FILTER_MIN_WAVELENGTH: float = 595e-09
FILTER_MAX_WAVELENGTH: float = 695e-09
FILTER_DEFAULT_RES: int = 24

def is_phoenix_installed(root: str) -> bool:
    """
    Check if "phoenix" is installed.

    Returns
    -------
    installed: bool
        True if all the phoenix files are present else false.
    """
    home: str = paths.concat([root, HOME])
    if not os.path.exists(home):
        return False

    for path in PATHS:
        file: str = paths.concat([home, path])
        if not os.path.isfile(file):
            return False

    return True

def make_phoenix_dirs(root: str) -> None:
    """
    Build the directory structure demanded by `pysynphot`.

    The directory structure that `pysynphot` needs is
    ```
    root/
        grid/
            phoenix/
                phoenixm00/
                phoenixp03/
    ```

    Parameters
    ----------
    root: str
        The directory in which to build.

    Examples
    --------
    >>> import os
    >>> os.mkdir("tmp")
    >>> make_phoenix_dirs("tmp")
    >>> os.path.exists("tmp/grid")
    ::: True
    >>> os.path.exists("tmp/grid/phoenix")
    ::: True
    >>> os.path.exists("tmp/grid/phoenix/phoenixm00")
    ::: True
    >>> os.path.exists("tmp/grid/phoenix/phoenixp03")
    ::: True
    """
    home: str = paths.concat([root, HOME])
    paths.mkdir_and_parents(home)

    for pnx in [M00, P03]:
        path: str = paths.concat([home, pnx])
        if not os.path.exists(path):
            os.mkdir(path)

def install_phoenix(root: str, /, full: bool = False) -> bool:
    """
    Install phoenix from the web.

    Parameters
    ----------
    root: str
        The directory to use for the install.
    full: bool
        True if the entire file is to be downloaded. 
        False if only the first byte is to be downloaded.
        False is used for testing purposes only.

    Examples
    --------
    >>> import os
    >>> os.mkdir("tmp")
    >>> install_phoenix("tmp", full = True)
    >>> is_phoenix_installed("tmp")
    ::: True
    """
    make_phoenix_dirs(root)

    if not full:
        warnings.warn("full = False")

    for file in PATHS:
        path: str = paths.concat([root, HOME, file])
        url: str = paths.concat([URL, file])

        print("Downloading: {}.".format(url))

        if os.path.isfile(path):
            warnings.warn("{} already exists.".format(path))
        else:
            if full:
                https.download_file_from_https(path, url)
            else: 
                https.download_byte_from_https(path, url)

def set_phoenix_environ(root: str) -> None:
    """
    Make sure that the phoenix environment variables are set.

    `pysynphot` requires that the environment variable `PYSYN_CDBS` is 
    set before use. This function checks if it is already set and 
    overrides it if it is, also printing a warning.

    Parameters
    ----------
    root: str
        The location to set in the phoenix environment.

    Examples
    --------
    >>> import os
    >>> os.mkdir("tmp")
    >>> os.environ["PYSYN_CDBS"] = "tmp"
    >>> set_phoenix_environment("tmp")
    ::: PYSYN_CDBS was set to: tmp
    """
    SYN: str = "PYSYN_CDBS"

    if os.environ.get(SYN) != None:
        warnings.warn("{} was set to: {}".format(SYN, os.environ.get(SYN)))

    os.environ[SYN] = root

# TODO: Document
def make_phoenix_spectra(root: str) -> float:
    """
    Generate the spectra using phoenix.

    The spectrum is returned in an array so that the leading axis
    can be indexed via the following convention.
    WAVES: int = 0
    ALPHA_CEN_A: int = 1
    ALPHA_CEN_B: int = 2

    Parameters
    ----------
    root: str
        The directory to look for the phoenix files in.
    """
    if not is_phoenix_installed(root):
        raise ValueError

    set_phoenix_environ(root)

    import pysynphot

    alpha_cen_a_spectrum: float = pysynphot.Icat(
        "phoenix",
        ALPHA_CEN_A_SURFACE_TEMP,
        ALPHA_CEN_A_METALICITY,
        ALPHA_CEN_A_SURFACE_GRAV,
    )

    alpha_cen_b_spectrum: float = pysynphot.Icat(
        "phoenix",
        ALPHA_CEN_B_SURFACE_TEMP,
        ALPHA_CEN_B_METALICITY,
        ALPHA_CEN_B_SURFACE_GRAV,
    )

    spectra: float = np.array([
            math.angstrom_to_m(alpha_cen_a_spectrum.wave),
            math.normalise(alpha_cen_a_spectrum.flux),
            math.normalise(alpha_cen_b_spectrum.flux),
        ], dtype=float)

    return spectra

def clip_phoenix_spectra(spectra: float) -> float:
    """
    Select the spectra within the filter.

    Parameters
    ----------
    spectra: float
        The spectra representing alpha centrauri across the entire 
        electromagnetic spectrum.

    Returns
    -------
    spectra: float
        The spectra within the filter. 
    """
    decision: bool = np.logical_and(
        (FILTER_MIN_WAVELENGTH < spectra[WAVES]),
        (spectra[WAVES] < FILTER_MAX_WAVELENGTH)
    )

    return spectra[:, decision]

def resample_phoenix_spectra(spectra: float, number_of_wavelengths: int) -> float:
    """
    Downsample the spectra.

    Reduce the sampling of the spectra to a number that can fit within 
    the device memory. The downsampling is done by taking averages.

    Parameters
    ----------
    spectra: float
        The full resolution spectra.
    number_of_wavelengths: int
        The number of wavelengths to downsample to.

    Returns
    -------
    spectra: float
        The spectra sampled by the number of wavelengths.
    """
    size: int = spectra[WAVES].size
    resample_size: int = size - size % number_of_wavelengths
    spectra: float = spectra[:, :resample_size]
    resample_by: int = resample_size // number_of_wavelengths 
    return math.downsample_along_axis(spectra, resample_by, axis=1)

def save_phoenix_spectra(root: str, spectra: str) -> None:
    """
    Simulate the spectrum of the alpha centauri binary using `pysynphot`.

    The output is saved to a file so that
    it can be used again later without having to be reloaded.

    Parameters
    ----------
    root: str
        The directory to save the file in.
    spectra: float
        An array representation of the spectra. This should be 3 by the 
        number of wavelengths.
    """
    file: str = paths.concat([root, "spectra.csv"])

    with open(file, "w") as fspectra:
        fspectra.write("alpha cen a waves (m), ")
        fspectra.write("alpha cen a flux (W/m/m), ")
        fspectra.write("alpha cen b flux (W/m/m)\n")

        for i in np.arange(spectra.shape[1], dtype=int):
            fspectra.write("{}, ".format(spectra[WAVES][i]))
            fspectra.write("{}, ".format(spectra[ALPHA_CEN_A][i]))
            fspectra.write("{}\n".format(spectra[ALPHA_CEN_B][i]))

def is_spectra_installed(root: str) -> bool:
    """
    Check if the spectra are installed.

    Parameters
    ----------
    root: str
        The directory to search for an installation in.

    Returns
    -------
    installed: bool
        True is the spectra are installed else false.

    Examples
    --------
    >>> import os
    >>> os.mkdir("tmp")
    >>> is_spectra_installed("tmp")
    ::: False
    >>> open("tmp/spectra.csv", "w").close()
    >>> is_spectra_installed("tmp")
    ::: True
    """
    if not os.path.isdir(root):
        return False
        
    return os.path.isfile(paths.concat([root, "spectra.csv"]))

def install_spectra(root: str, number_of_wavelengths: int) -> None:
    """
    Make and save the alpha centauri spectrum at a given resolution.

    This function assumes that phoenix is already installed. If it is 
    not installed then use the `install_phoenix` command. This module 
    is not necessarily intended to be user facing so be aware that in 
    general root should be TOLIMAN_HOME, but that is enforced upstream.

    Parameters
    ----------
    root: str
        The directory to install the spectrum.
    number_of_wavelengths: int
        The resolution of the spectrum. < 25 is not recommended but this
        is not enforced.

    Examples
    --------
    >>> import os
    >>> import shutil
    >>> if os.path.isdir("tmp"):
    ...     shutil.rmtree("tmp")
    >>> is_phoenix_installed("tmp")
    ::: False
    >>> install_phoenix("tmp")
    >>> is_phoenix_installed("tmp")
    ::: True
    >>> install_spectra("tmp")
    >>> is_spectra_installed("tmp")
    ::: True
    """
    if not is_phoenix_installed(root):
        raise ValueError("Phoenix not installed!")

    spectra: float = resample_phoenix_spectra(
        clip_phoenix_spectra(make_phoenix_spectra(root)), 
        number_of_wavelengths
    )

    save_phoenix_spectra(root, spectra)
    

    
