import os
import warnings
import tqdm
import requests

__author__ = "Jordan Dennis"

HOME: str = "toliman/assets/grid/phoenix"
PHOENIX_HOME: str = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/phoenix"
MASK_HOME: str = "https://github.com/ConaireD/TolimanWIP/raw/mask/mask.npy"
PHOENIXM00: str = "phoenixm00"
PHOENIXP03: str = "phoenixp03"
PHOENIX_PATHS: str = [
    "{}/{}_5200.fits".format(PHOENIXM00, PHOENIXM00),
    "{}/{}_5300.fits".format(PHOENIXM00, PHOENIXM00),
    "{}/{}_5700.fits".format(PHOENIXM00, PHOENIXM00),
    "{}/{}_5800.fits".format(PHOENIXM00, PHOENIXM00),
    "{}/{}_5200.fits".format(PHOENIXP03, PHOENIXP03),
    "{}/{}_5300.fits".format(PHOENIXP03, PHOENIXP03),
    "{}/{}_5700.fits".format(PHOENIXP03, PHOENIXP03),
    "{}/{}_5800.fits".format(PHOENIXP03, PHOENIXP03),
    "catalog.fits"
]

def _is_phoenix_installed() -> bool:
    """
    Check if "phoenix" is installed.

    Returns
    -------
    installed: bool
        True if all the phoenix files are present else false.
    """
    if not os.path.exists(HOME):
        return False

    for path in PHOENIX_PATHS:
        rel_path: str = "{}/{}".format(HOME, path)
        if not os.path.isfile(path):
            return False

    return True

def _accumulate_path(strings: list, paths: str = []) -> list:
    """
    Incrementally build a path from a list.

    Parameters
    ----------
    strings: list 
        A list of directories that can be pasted together to form a 
        list.
    paths: list
        A list of the paths to each dir.

    Returns
    -------
    paths: list
        Starting with the root directory, which is assumed to be the 
        first entry in `strings`, a list of directories growing from
        root.

    Examples
    --------
    >>> strings: list = ["root", "dev", "null"]
    >>> _accumulate_path(strings)
    ::: ["root", "root/dev", "root/dev/null"]
    """
    if not strings:
        return paths
    else:
        if not paths:
            paths.append(strings.pop(0))
        else:
            paths.append(paths[-1] + "/" + strings.pop(0))
        return _accumulate_path(strings, paths)

def _is_mask_installed() -> bool:
    return os.path.isfile("{}/mask.npy".format(MASK)

def _install_mask() -> bool:
    """
    Install phoenix from the web.
    """
    MASK: str = "toliman/assets"
    if not os.path.exists(MASK):
        for path in _accumulate_path(MASK.split("/")):
            if not os.path.exists(path):
                os.mkdir(path)

    with open("{}/mask.npy".format(MASK), "wb") as file_dev:
        response: iter = requests.get(MASK_HOME, stream=True)
        total_size: int = int(response.headers.get('content-length', 0))

        print("Downloading: {}.".format(MASK_HOME))

        progress: object = tqdm.tqdm(
            total=total_size,
            unit='iB', 
            unit_scale=True
        )

        for data in response.iter_content(1024):
            progress.update(len(data))
            file_dev.write(data)

def _install_phoenix() -> bool:
    """
    Install the mask from the web.
    """
    if not os.path.exists(HOME):
        for path in _accumulate_path(HOME.split("/")):
            if not os.path.exists(path):
                os.mkdir(path)

    for path in [PHOENIXM00, PHOENIXP03]:
        rel_path: str = "{}/{}".format(HOME, path)
        if not os.path.exists(rel_path):
            os.mkdir(rel_path)

    for file in PHOENIX_PATHS:
        path: str = "{}/{}".format(HOME, file)
    
        if not os.path.isfile(path):
            url: str = "{}/{}".format(PHOENIX_HOME, file)

            with open(path, "wb") as file_dev:
                url: str = "{}/{}".format(PHOENIX_HOME, file)
                response: iter = requests.get(url, stream=True)
                total_size: int = int(response.headers.get('content-length', 0))

                print("Downloading: {}.".format(url))

                progress: object = tqdm.tqdm(
                    total=total_size,
                    unit='iB', 
                    unit_scale=True
                )

                for data in response.iter_content(1024):
                    progress.update(len(data))
                    file_dev.write(data)

def _simulate_alpha_cen_spectra(number_of_wavelengths: int = 25) -> None:
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

def _simulate_background_stars() -> None:
    """
    Sample the Gaia database for typical background stars.

    The primary use of this function is to
    build a sample that can be used to look for biases.
    """
    from astroquery.gaia import Gaia

    conical_query = """
    SELECT
        TOP 12000 
        ra, dec, phot_g_mean_flux AS flux
    FROM
        gaiadr3.gaia_source
    WHERE
        CONTAINS(POINT('', ra, dec), CIRCLE('', {}, {}, {})) = 1 AND
        phot_g_mean_flux IS NOT NULL
    """

    bg_ra: float = 220.002540961 + 0.1
    bg_dec: float = -60.8330381775
    alpha_cen_flux: float = 1145.4129625806625
    bg_win: float = 2.0 / 60.0
    bg_rad: float = 2.0 / 60.0 * np.sqrt(2.0)

    bg_stars: object = Gaia.launch_job(conical_query.format(bg_ra, bg_dec, bg_rad))

    bg_stars_ra: float = np.array(bg_stars.results["ra"]) - bg_ra
    bg_stars_dec: float = np.array(bg_stars.results["dec"]) - bg_dec
    bg_stars_flux: float = np.array(bg_stars.results["flux"])

    ra_in_range: float = np.abs(bg_stars_ra) < bg_win
    dec_in_range: float = np.abs(bg_stars_dec) < bg_win
    in_range: float = ra_in_range & dec_in_range
    sample_len: float = in_range.sum()

    bg_stars_ra_crop: float = bg_stars_ra[in_range]
    bg_stars_dec_crop: float = bg_stars_dec[in_range]
    bg_stars_flux_crop: float = bg_stars_flux[in_range]
    bg_stars_rel_flux_crop: float = bg_stars_flux_crop / alpha_cen_flux

    print(sample_len)

    with open("toliman/assets/background.csv", "w") as sheet:
        sheet.write("ra,dec,rel_flux\n")
        for row in np.arange(sample_len):
            sheet.write(f"{bg_stars_ra_crop[row]},")
            sheet.write(f"{bg_stars_dec_crop[row]},")
            sheet.write(f"{bg_stars_rel_flux_crop[row]}\n")

def _normalise(arr: float) -> float:
    """
    Rescale and array onto [0, 1].

    Parameters
    ----------
    arr: float
        Any array.

    Returns
    -------
    arr: float
        An array of floating point numbers over the range [0, 1].
    """
    return (arr - arr.min()) / arr.ptp()


def main():
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



main()
