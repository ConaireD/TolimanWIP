import os
import paths
import https
import pysynphot

__author__ = "Jordan Dennis"
__all__ = [
    "is_phoenix_installed",
    "install_phoenix",
    "make_phoenix_dirs",
    "make_phoenix_spectra",
    "save_phoenix_spectra",
]

HOME: str = "grid/phoenix"
URL: str = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/phoenix"
M00: str = "phoenixm00"
P03: str = "phoenixp03"
NUMS: list = [5200, 5300, 5700, 5900]
PATHS: str = ["catalog.fits"] + [
    "{}/{}_{}.fits".format(HOME, phoenix_t, phoenix_t, num) 
        for num in NUMS 
        for phoenix_t in [M00, P03]
]

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
    if not os.path.exists(home):
        for path in paths.accumulate(home.split("/")):
            if not os.path.exists(path):
                os.mkdir(path)

    paths: list = [
        paths.concat(home, pnx) 
            for pnx in [M00, P03]
    ]

    for path in paths:
        if not os.path.exists(rel_path):
            os.mkdir(rel_path)

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

    for file in PATHS:
        path: str = paths.concat([home, file])
        url: str = paths.concat([URL, file])

        print("Downloading: {}.".format(url))

        if full:
            download_file_from_https(file, url)
        else: 
            warning.warn("full = False")
            download_byte_from_https(file, url)

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

    if not os.environ.get(SYN):
        warnings.warn("{} was set to: {}".format(SYN, os.environ.get(SYN)))

    os.environ[SYN] = root

def make_phoenix_spectra(root: str, number_of_wavelengths: int) -> float:
    set_phoenix_environment(root)

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

    WAVES: int = 0
    ALPHA_CEN_A: int = 1
    ALPHA_CEN_B: int = 2

    spectra: float = np.array([
            math.angstrom_to_m(alpha_cen_a_spectrum.wave),
            math.normalise(alpha_cen_a_spectrum.flux),
            math.normalise(alpha_cen_b_spectrum.flux),
        ], dtype=float)

    return spectra


def save_phoenix_spectra(number_of_wavelengths: int = 25) -> None:
    """
    Simulate the spectrum of the alpha centauri binary using `pysynphot`.

    The output is saved to a file so that
    it can be used again later without having to be reloaded.

    Parameters
    ----------
    number_of_wavelengts: int
        The number of wavelengths that you wish to use for the simulation.
        The are taken from the `pysynphot` output by binning.
    """


    def angstrom_to_m(angstrom: float) -> float:
        m_per_angstrom: float = 1e-10
        return m_per_angstrom * angstrom

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

    WAVES: int = 0
    ALPHA_CEN_A: int = 1
    ALPHA_CEN_B: int = 2

    spectra: float = np.array([
            angstrom_to_m(alpha_cen_a_spectrum.wave),
            math.normalise(alpha_cen_a_spectrum.flux),
            math.normalise(alpha_cen_b_spectrum.flux),
        ], dtype=float)

    del alpha_cen_a_spectrum, alpha_cen_b_spectrum

    decision: bool = np.logical_and(
        (FILTER_MIN_WAVELENGTH < spectra[WAVES]),
        (spectra[WAVES] < FILTER_MAX_WAVELENGTH)
    )

    spectra: float = spectra[:, decision]

    del decision

    size: int = spectra[WAVES].size
    resample_size: int = size - size % number_of_wavelengths
    spectra: float = spectra[:, :resample_size]
    resample_by: int = resample_size // number_of_wavelengths 
    spectra: float = _downsample_along_axis(spectra, resample_by, axis=1)

    with open("toliman/assets/spectra.csv", "w") as fspectra:
        fspectra.write("alpha cen a waves (m), ")
        fspectra.write("alpha cen a flux (W/m/m), ")
        fspectra.write("alpha cen b flux (W/m/m)\n")

        for i in np.arange(number_of_wavelengths, dtype=int):
            fspectra.write("{}, ".format(spectra[WAVES][i]))
            fspectra.write("{}, ".format(spectra[ALPHA_CEN_A][i]))
            fspectra.write("{}\n".format(spectra[ALPHA_CEN_B][i]))

