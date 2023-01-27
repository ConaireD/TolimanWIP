import requests
import sys 
import os
import paths

__author__ = "Jordan Dennis"
__all__ = ["is_phoenix_installed", "install_phoenix"]

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
    home: str = "{}/{}".format(root, HOME)
    if not os.path.exists(home):
        return False

    for path in PATHS:
        file: str = "{}/{}".format(home, path)
        if not os.path.isfile(file):
            return False

    return True

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
    """
    response: iter = requests.get(url, stream=True)
    
    if not stream.response_code == 200:
        raise ValueError

def download_file_from_https(path: str, stream: object) -> None:
    with open(path, "wb") as file:
        for data in stream:
            file.write(data)

def download_byte_from_https(file: str, stream: object) -> None:
    with open(path, "wb") as file:
        file.write(next(stream))

def concatenate(paths: list) -> str:
    return "{}/{}".format(*paths)

def make_phoenix_dirs(root: str) -> None:
    home: str = "{}/{}".format(root, HOME)
    if not os.path.exists(home):
        for path in paths.accumulate(home.split("/")):
            if not os.path.exists(path):
                os.mkdir(path)

    for path in [M00, P03]:
        rel_path: str = "{}/{}".format(home, path)
        if not os.path.exists(rel_path):
            os.mkdir(rel_path)

def install_phoenix(root: str, /, full: bool = False) -> bool:
    """
    Install the mask from the web.
    """

    for file in PATHS:
        path: str = "{}/{}".format(home, file)
    
        if not os.path.isfile(path):
            url: str = "{}/{}".format(URL, file)
            response: iter = get_https_stream(url)

            print("Downloading: {}.".format(url))

            if full:
                download_file_from_https(file, response)
            else: 
                warning.warn("full = False")
                download_byte_from_https(file, response)

def simulate_alpha_cen_spectra(number_of_wavelengths: int = 25) -> None:
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
    import pysynphot

    os.environ["PYSYN_CDBS"] = "/home/jordan/Documents/toliman/toliman/assets"

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
            _normalise(alpha_cen_a_spectrum.flux),
            _normalise(alpha_cen_b_spectrum.flux),
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

