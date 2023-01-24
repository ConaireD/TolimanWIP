import jax.numpy as np
import jax
import dLux as dl
import equinox as eqx
import abc
import os

__author__ = "Jordan Dennis"
__all__ = [
    "TolimanDetector",
    "TolimanOptics",
    "AlphaCentauri",
    "Background",
    "_contains_instance",
    "_simulate_alpha_cen_spectra",
    "_simulate_background_stars",
    "_read_csv_to_jax_array",
]

DEFAULT_PUPIL_NPIX: int = 256
DEFAULT_DETECTOR_NPIX: int = 128
DEFAULT_NUMBER_OF_ZERNIKES: int = 5

DEFAULT_MASK_DIR: str = "/home/jordan/Documents/toliman/toliman/assets/mask.npy"
SPECTRUM_DIR: str = "/home/jordan/Documents/toliman/toliman/assets/spectra.csv"
BACKGROUND_DIR: str = "toliman/assets/background.csv"

TOLIMAN_PRIMARY_APERTURE_DIAMETER: float = 0.13
TOLIMAN_SECONDARY_MIRROR_DIAMETER: float = 0.032
TOLIMAN_DETECTOR_PIXEL_SIZE: float = dl.utils.arcseconds_to_radians(0.375)
TOLIMAN_WIDTH_OF_STRUTS: float = 0.01
TOLIMAN_NUMBER_OF_STRUTS: int = 3

DEFAULT_DETECTOR_JITTER: float = 2.0
DEFAULT_DETECTOR_SATURATION: float = 2500
DEFAULT_DETECTOR_THRESHOLD: float = 0.05

ALPHA_CENTAURI_SEPARATION: float = dl.utils.arcseconds_to_radians(8.0)
ALPHA_CENTAURI_POSITION: float = np.array([0.0, 0.0], dtype=float)
ALPHA_CENTAURI_MEAN_FLUX: float = 1.0
ALPHA_CENTAURI_CONTRAST: float = 2.0
ALPHA_CENTAURI_POSITION_ANGLE: float = 0.0

ALPHA_CEN_A_SURFACE_TEMP: float = 5790.0
ALPHA_CEN_A_METALICITY: float = 0.2
ALPHA_CEN_A_SURFACE_GRAV: float = 4.0

ALPHA_CEN_B_SURFACE_TEMP: float = 5260.0
ALPHA_CEN_B_METALICITY: float = 0.23
ALPHA_CEN_B_SURFACE_GRAV: float = 4.37

FILTER_MIN_WAVELENGTH: float = 595e-09
FILTER_MAX_WAVELENGTH: float = 695e-09
FILTER_DEFAULT_RES: int = 24


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

def _read_csv_to_jax_array(_file_name: str) -> float:
    """
    Read a CSV using `jax`.

    This is a private function and following convention it assumes that the 
    file exists. There is no error checking!

    Parameters
    ----------
    _file_name: str
        The name of the file to read.

    Returns
    -------
    arr: float
        The information in the CSV. The headings are not returned and so it 
        is up to you to keep track of what each column is.
    """
    with open(_file_name, "r") as file:
        lines: list = file.readlines()
        _: str = lines.pop(0)
        strip: callable = lambda _str: _str.strip().split(",")
        str_to_float: callable = lambda _str: float(_str.strip())
        entries: list = jax.tree_map(strip, lines)
        _file: float = jax.tree_map(str_to_float, entries)

    return np.array(_file)

def _contains_instance(_list: list, _type: type) -> bool:
    """
    Check to see if a list constains an element of a certain type.

    Parameters
    ----------
    _list: list
        The list to search.
    _type: type
        The type to check for.

    Returns
    -------
    contains: bool
        True if _type was found else False.
    """
    if _list:
        for _elem in _list:
            if isinstance(_elem, _type):
                return True
    return False


def _downsample_square_grid(arr: float, m: int) -> float:
    """
    Resample a square array by a factor of `m`.

    Parameters
    ----------
    arr: float
        An `NxN` array.
    m: float
        The factor to downsample by so that the final shape is `(N/m)x(N/m)`.
        This implies that `N % m == 0`.

    Examples
    --------
    ```python
    >>> import jax.numpy as np
    >>> up_arr: float = np.ones((1024, 1024), dtype=float)
    >>> down_arr: float = _downsample(arr, 4)
    >>> down_arr.shape
    (256, 256)
    ```
    """
    n: int = arr.shape[0]
    out: int = n // m

    dim_one: float = arr.reshape((n * out, m)).sum(1).reshape(n, out).T
    dim_two: float = dim_one.reshape((out * out, m)).sum(1).reshape(out, out).T

    return dim_two / m / m


def _downsample_along_axis(arr: float, m: int, axis: int = 0) -> float:
    """
    Resampling an array by averaging along a particular dimension.

    Parameters
    ----------
    arr: float
        The array to resample.
    m: int
        The factor by which to downsample the axis.
    axis: int = 0
        The axis to resample.

    Returns
    -------
    arr: float
        The resampled array
    """
    shape: tuple = arr.shape
    n: int = shape[axis]
    out: int = n // m
    new: tuple = tuple([out if i == axis else dim for i, dim in enumerate(shape)] + [m])
    return arr.reshape(new).sum(-1) / m


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


# TODO: Add arguments
# TODO: optimise
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


def _simulate_data(model: object, scale: float) -> float:
    """
    Simulate some fake sata for comparison.

    Parameters
    ----------
    model: object
        A model of the toliman. Should inherit from `dl.Instrument` or
        be an instance.
    scale: float
        How noisy is the detector?

    Returns
    -------
    data: float, photons
        A noisy psf.
    """
    psf: float = model.model()
    noisy_psf: float = photon_noise(psf)
    noisy_image: float = noisy_psf + latent_detector_noise(scale, psf.shape)
    return noisy_image


def pixel_response(shape: float, threshold: float, seed: int = 1) -> float:
    """
    Simulate pixel reponses.

    Parameters
    ----------
    shape: tuple[int]
        The array shape to populate with a random pixel response.
    threshold: float
        How far from 1. does the pixel response typically vary.
    seed: int = 1
        The seed of the random generation.

    Returns
    -------
    pixel_response: float
        An array of the pixel responses.
    """
    key: object = jax.random.PRNGKey(seed)
    return 1.0 + threshold * jax.random.normal(key, shape)


def photon_noise(psf: float, seed: int = 0) -> float:
    """
    Simulate photon noise.

    Parameters
    ----------
    psf: float
        The psf on which to add photon noise.
    seed: int = 1
        The seed of the random generation.

    Returns
    -------
    photon_noise: float
        A noisy psf.
    """
    key = jax.random.PRNGKey(seed)
    return jax.random.poisson(key, psf)


def latent_detector_noise(scale: float, shape: float, seed: int = 0) -> float:
    """
    Simulate some gaussian latent noise.

    Parameters
    ----------
    scale: float, photons
        The standard deviation of the gaussian in photons.
    shape: tuple
        The shape of the array to generate the noise on.
    seed: int = 0
        The seed of the random generation.

    Returns
    -------
    det_noise: float, photons
        The an additional noise source from the detector.
    """
    key: object = jax.random.PRNGKey(seed)
    return scale * jax.random.normal(key, shape)


MASK_TOO_LARGE_ERR_MSG = """ 
The mask you have loaded had a higher resolution than the pupil. 
A method of resolving this has not yet been created. Either 
change the value of the `DEFAULT_PUPIL_NPIX` constant or use a 
different mask.
"""

MASK_SAMPLING_ERR_MSG = """
The mask you have loaded could not be downsampled onto the pupil. 
The shape of the mask was ({%i, %i}), but ({%i, %i}) was expected.
Either change the value of the environment variable 
`DEFAULT_PUPIL_NPIX` or use a different mask.
"""

MASK_IMPORT_ERR_MSG = """
The file address that of the mask did not exist. Make suer that 
you have a `.npy` representation of the phase mask available 
and have provided the correct file address to the constructor.
"""

FRESNEL_USE_ERR_MSG = """
You have request operation in Fresenl mode. This has not currently 
been implemented. Once implemented it is not recommended that you 
use the feature as it is very slow and the zernike terms should 
be sufficient for most purposes.
"""

POLISH_USE_ERR_MSG = """
You have requested that the mirror polish be simulated this has 
not yet been implemented although it is planned in an upcoming 
release.
"""

DETECTOR_EMPTY_ERR_MSG = """
You have provided no detector layers and not asked for any of the 
defaults. This implies that the detector does not contain any 
layers which is not a valid state. If you do not wish to model 
any detector effects do not provide a detector in construction.
"""

DETECTOR_REPEATED_ERR_MSG = """
You have provided a layer that is also a default layer. Make sure 
that each type of detector is only provided once. 
"""


class CollectionInterface(abc.ABC):
    @abc.abstractmethod
    def to_optics_list(self: object) -> list:
        """
        Get the optical elements that make up the object as a list.

        Returns
        -------
        optics: list
            The optical layers in order in a list.
        """

    @abc.abstractmethod
    def insert(self: object, optic: object, index: int) -> object:
        """
        Add an additional layer to the optical system.

        Parameters
        ----------
        optic: object
            A `dLux.OpticalLayer` to include in the model.
        index: int
            Where in the list of layers to add optic.

        Returns
        -------
        toliman: TolimanOptics
            A new `TolimanOptics` instance with the applied update.
        """

    @abc.abstractmethod
    def remove(self: object, index: int) -> object:
        """
        Take a layer from the optical system.

        Parameters
        ----------
        index: int
            Where in the list of layers to remove an optic.

        Returns
        -------
        toliman: TolimanOptics
            A new `TolimanOptics` instance with the applied update.
        """

    @abc.abstractmethod
    def append(self: object, optic: object) -> object:
        """
        Place a new optic at the end of the optical system.

        Parameters
        ----------
        optic: object
            The optic to include. It must be a subclass of the
            `dLux.OpticalLayer`.

        Returns
        -------
        optics: object
            The new optical system.
        """

    @abc.abstractmethod
    def pop(self: object) -> object:
        """
        Remove the last element in the optical system.

        Please note that this differs from the `.pop` method of the
        `list` class because it does not return the popped element.

        Returns
        -------
        optics: object
            The optical system with the layer removed.
        """


# TODO: I need to work out how to do the regularisation internally so
#       that the values which are returned are always correct.
class TolimanOptics(dl.Optics, CollectionInterface):
    """
    Simulates the optical system of the TOLIMAN telescope.

    It is designed to occupy the `optics` kwarg of `dl.Instrument`.
    The `TolimanOptics` provides a default implementation that
    can be extended using the `.add` method. There are also
    several ways that the `TolimanOptics` can be initialised.

    Examples
    --------
    ```
    >>> toliman_optics: object = TolimanOptics()
    >>> toliman_optics: object = TolimanOptics(simulate_aberrations = False)
    >>> toliman_optics: object = TolimanOptics(pixels_in_pupil = 1024)
    ```

    For more options run `help(TolimanOptics.__init__)`.
    """

    def __init__(
        self: object,
        simulate_polish: bool = False,
        simulate_aberrations: bool = True,
        operate_in_fresnel_mode: bool = False,
        operate_in_static_mode: bool = True,
        number_of_zernikes: int = DEFAULT_NUMBER_OF_ZERNIKES,
        pixels_in_pupil: int = DEFAULT_PUPIL_NPIX,
        pixels_on_detector: int = DEFAULT_DETECTOR_NPIX,
        path_to_mask: str = DEFAULT_MASK_DIR,
        path_to_filter: str = "assets/filter.npy",
        path_to_polish: str = "assets/polish.npy",
    ) -> object:
        """
        Simulate the Toliman telescope.

        Parameters
        ----------
        simulate_polish: bool = True
            True if a layer should be included simulating the polish
            on the secondary mirror.
        simulate_aberrations: bool = True
            True if the aberrations should be included.
        operate_in_fresnel_mode: bool = False
            True if the simulation should use Fresnel instead of
            Fourier optics.
        operate_in_static_mode: bool = True
            True if the pupil of the aperture should be modelled
            as static. This will improve performance so only change
            it if you want to learn a parameter of the aperture.
        number_of_zernikes: int = DEFAULT_NUMBER_OF_ZERNIKES
            The number of zernike polynomials that should be used
            to model the aberrations.
        pixels_in_pupil: int = DEFAULT_PUPIL_NPIX
            The number of pixels in the pupil plane.
        pixels_on_detector: int = DEFAULT_DETECTOR_NPIX
            The number of pixels in the detector plane.
        path_to_mask: str = "assets/mask.npy"
            The file location of a `.npy` file that contains an
            array representation o the mask.
        path_to_filter: str = "assets/filter.npy"
            The file location of a `.npy` file that contains an
            array representation og the filter.
        path_to_polish: str = "assets/polish.npy"
            The file location of a `.npy` file that contains an
            array representation of the secondary mirror polish.
        """
        toliman_layers: list = [
            dl.CreateWavefront(
                pixels_in_pupil,
                TOLIMAN_PRIMARY_APERTURE_DIAMETER,
                wavefront_type="Angular",
            )
        ]

        # Adding the pupil
        dyn_toliman_pupil: object = dl.CompoundAperture(
            [
                dl.UniformSpider(TOLIMAN_NUMBER_OF_STRUTS, TOLIMAN_WIDTH_OF_STRUTS),
                dl.AnnularAperture(
                    TOLIMAN_PRIMARY_APERTURE_DIAMETER / 2.0,
                    TOLIMAN_SECONDARY_MIRROR_DIAMETER / 2.0,
                ),
            ]
        )

        if operate_in_static_mode:
            static_toliman_pupil: object = dl.StaticAperture(
                dyn_toliman_pupil,
                npixels=pixels_in_pupil,
                diameter=TOLIMAN_PRIMARY_APERTURE_DIAMETER,
            )

            toliman_layers.append(static_toliman_pupil)
        else:
            toliman_layers.append(dyn_toliman_pupil)

        # Loading the mask.
        try:
            loaded_mask: float = np.load(path_to_mask)
            loaded_shape: tuple = loaded_mask.shape
            loaded_width: int = loaded_shape[0]

            mask: float
            if not loaded_width == pixels_in_pupil:
                if loaded_width < pixels_in_pupil:
                    raise NotImplementedError(MASK_TOO_LARGE_ERR_MSG)
                if loaded_width % pixels_in_pupil == 0:
                    downsample_by: int = loaded_width // pixels_in_pupil
                    mask: float = _downsample_square_grid(loaded_mask, downsample_by)
                else:
                    raise ValueError(MASK_SAMPLING_ERR_MSG)
            else:
                mask: float = loaded_mask

            del loaded_mask
            del loaded_shape
            del loaded_width
        except IOError as ioe:
            raise ValueError(MASK_IMPORT_ERR_MSG)

        toliman_layers.append(dl.AddOPD(mask))

        # Generating the Zernikes
        # TODO: Make zernike_coefficients a function
        if simulate_aberrations:
            nolls: list = np.arange(2, number_of_zernikes + 2, dtype=int)
            seed: object = jax.random.PRNGKey(0)
            coeffs: list = 1e-08 * jax.random.normal(seed, (number_of_zernikes,))

            toliman_aberrations: object = dl.StaticAberratedAperture(
                dl.AberratedAperture(
                    noll_inds=nolls,
                    coefficients=coeffs,
                    aperture=dl.CircularAperture(
                        TOLIMAN_PRIMARY_APERTURE_DIAMETER / 2.0
                    ),
                ),
                npixels=pixels_in_pupil,
                diameter=TOLIMAN_PRIMARY_APERTURE_DIAMETER,
            )

            toliman_layers.append(toliman_aberrations)

        toliman_layers.append(dl.NormaliseWavefront())

        if simulate_polish:
            raise NotImplementedError(POLISH_USE_ERR_MSG)

        # Adding the propagator
        toliman_body: object
        if not operate_in_fresnel_mode:
            toliman_body: object = dl.AngularMFT(
                pixels_on_detector, TOLIMAN_DETECTOR_PIXEL_SIZE
            )
        else:
            raise NotImplementedError(FRESNEL_USE_ERR_MSG)

        toliman_layers.append(toliman_body)

        # Renormalising the flux.
        toliman_layers.append(dl.NormaliseWavefront())

        super().__init__(layers=toliman_layers)

    def to_optics_list(self: object) -> list:
        """
        Get the optical elements that make up the object as a list.

        Returns
        -------
        optics: list
            The optical layers in order in a list.
        """
        return list(self.layers.values())

    def insert(self: object, optic: object, index: int) -> object:
        """
        Add an additional layer to the optical system.

        Parameters
        ----------
        optic: object
            A `dLux.OpticalLayer` to include in the model.
        index: int
            Where in the list of layers to add optic.

        Returns
        -------
        toliman: TolimanOptics
            A new `TolimanOptics` instance with the applied update.
        """
        correct_type: bool = False
        if isinstance(optic, dl.optics.OpticalLayer):
            correct_type: bool = True
        elif isinstance(optic, dl.apertures.ApertureLayer):
            correct_type: bool = True

        if not correct_type:
            raise ValueError("Inserted optics must be optical layers.")

        if index < 0:
            raise ValueError("`index` must be positive.")

        new_layers: list = self.to_optics_list()
        _: None = new_layers.insert(index, optic)
        dl_new_layers: dict = dl.utils.list_to_dictionary(new_layers)
        return eqx.tree_at(lambda x: x.layers, self, dl_new_layers)

    def remove(self: object, index: int) -> object:
        """
        Take a layer from the optical system.

        Parameters
        ----------
        index: int
            Where in the list of layers to remove an optic.

        Returns
        -------
        toliman: TolimanOptics
            A new `TolimanOptics` instance with the applied update.
        """
        if index < 0:
            raise ValueError("`index` must be positive.")

        new_layers: list = self.to_optics_list()
        length: int = len(new_layers)

        if index > length:
            raise ValueError("`index` must be within the optical system.")

        _: None = new_layers.pop(index)
        dl_new_layers: dict = dl.utils.list_to_dictionary(new_layers)
        return eqx.tree_at(lambda x: x.layers, self, dl_new_layers)

    def append(self: object, optic: object) -> object:
        """
        Place a new optic at the end of the optical system.

        Parameters
        ----------
        optic: object
            The optic to include. It must be a subclass of the
            `dLux.OpticalLayer`.

        Returns
        -------
        optics: object
            The new optical system.
        """
        correct_type: bool = False
        if isinstance(optic, dl.optics.OpticalLayer):
            correct_type: bool = True
        elif isinstance(optic, dl.apertures.ApertureLayer):
            correct_type: bool = True

        if not correct_type:
            raise ValueError("Inserted optics must be optical layers.")

        new_layers: list = self.to_optics_list()
        _: None = new_layers.append(optic)
        dl_new_layers: dict = dl.utils.list_to_dictionary(new_layers)
        return eqx.tree_at(lambda x: x.layers, self, dl_new_layers)

    def pop(self: object) -> object:
        """
        Remove the last element in the optical system.

        Please note that this differs from the `.pop` method of
        the `list` class  because it does not return the popped element.

        Returns
        -------
        optics: object
            The optical system with the layer removed.
        """
        new_layers: list = self.to_optics_list()
        _: object = new_layers.pop()
        dl_new_layers: dict = dl.utils.list_to_dictionary(new_layers)
        return eqx.tree_at(lambda x: x.layers, self, dl_new_layers)


class TolimanDetector(dl.Detector, CollectionInterface):
    """
    Represents the Toliman detector.

    A default implementation of a generic detector that is designed
    to be used with the `dLux.Instrument`.

    Examples
    --------
    ```py
    >>> toliman_detector: object = TolimanDetector()
    >>> toliman_detector: object = TolimanDetector(simulate_jitter = False)
    ```
    """

    def __init__(
        self: object,
        simulate_jitter: bool = True,
        simulate_pixel_response: bool = True,
        simulate_saturation: bool = True,
        extra_detector_layers: list = [],
    ) -> object:
        """
        Simulate the Toliman detector.

        Parameters
        ----------
        simulate_jitter: bool = True
            True if jitter should be included in the simulation of the
            detector.
        simulate_pixel_response: bool = True
            True if a pixel response should be included in the simulation
            of the detector.
        simulate_saturation: bool = True
            True if staturation should be included in the simulation of
            the detector.
        extra_detector_layers: list = []
            Extra detector effects besides the default ones.
        """
        detector_layers: list = []

        if simulate_jitter:
            detector_layers.append(dl.ApplyJitter(DEFAULT_DETECTOR_JITTER))

            # TODO: Make a contains instance function
            if _contains_instance(extra_detector_layers, dl.ApplyJitter):
                raise ValueError(DETECTOR_REPEATED_ERR_MSG)

        if simulate_saturation:
            detector_layers.append(dl.ApplySaturation(DEFAULT_DETECTOR_SATURATION))

            if _contains_instance(extra_detector_layers, dl.ApplySaturation):
                raise ValueError(DETECTOR_REPEATED_ERR_MSG)

        if simulate_pixel_response:
            detector_layers.append(
                dl.ApplyPixelResponse(
                    pixel_response(
                        (DEFAULT_DETECTOR_NPIX, DEFAULT_DETECTOR_NPIX),
                        DEFAULT_DETECTOR_THRESHOLD,
                    )
                )
            )

            if _contains_instance(extra_detector_layers, dl.ApplyPixelResponse):
                raise ValueError(DETECTOR_REPEATED_ERR_MSG)

        detector_layers.extend(extra_detector_layers)

        if detector_layers:
            super().__init__(detector_layers)
        else:
            raise ValueError(DETECTOR_EMPTY_ERR_MSG)

    def to_optics_list(self: object) -> list:
        """
        Get the optical elements that make up the object as a list.

        Returns
        -------
        optics: list
            The optical layers in order in a list.
        """
        return list(self.layers.values())

    def insert(self: object, optic: object, index: int) -> object:
        """
        Add an additional layer to the optical system.

        Parameters
        ----------
        optic: object
            A `dLux.OpticalLayer` to include in the model.
        index: int
            Where in the list of layers to add optic.

        Returns
        -------
        toliman: TolimanOptics
            A new `TolimanOptics` instance with the applied update.
        """
        if not isinstance(optic, dl.detectors.DetectorLayer):
            raise ValueError("Inserted optics must be optical layers.")

        if index < 0:
            raise ValueError("`index` must be positive.")

        new_layers: list = self.to_optics_list()
        length: int = len(new_layers)

        if index > length:
            raise ValueError("`index` is outside the layers.")
        
        _: None = new_layers.insert(index, optic)
        dl_new_layers: dict = dl.utils.list_to_dictionary(new_layers)
        return eqx.tree_at(lambda x: x.layers, self, dl_new_layers)

    def remove(self: object, index: int) -> object:
        """
        Take a layer from the optical system.

        Parameters
        ----------
        index: int
            Where in the list of layers to remove an optic.

        Returns
        -------
        toliman: TolimanOptics
            A new `TolimanOptics` instance with the applied update.
        """
        if index < 0:
            raise ValueError("`index` must be positive.")

        new_layers: list = self.to_optics_list()
        length: int = len(new_layers)

        if index > length:
            raise ValueError("`index` must be within the detector.")

        _: None = new_layers.pop(index)
        dl_new_layers: dict = dl.utils.list_to_dictionary(new_layers)
        return eqx.tree_at(lambda x: x.layers, self, dl_new_layers)

    def append(self: object, optic: object) -> object:
        """
        Place a new optic at the end of the optical system.

        Parameters
        ----------
        optic: object
            The optic to include. It must be a subclass of the
            `dLux.OpticalLayer`.

        Returns
        -------
        optics: object
            The new optical system.
        """
        if not isinstance(optic, dl.detectors.DetectorLayer):
            raise ValueError("Inserted optics must be a detector layer.")

        new_layers: list = self.to_optics_list()
        _: None = new_layers.append(optic)
        dl_new_layers: dict = dl.utils.list_to_dictionary(new_layers)
        return eqx.tree_at(lambda x: x.layers, self, dl_new_layers)

    def pop(self: object) -> object:
        """
        Remove the last element in the optical system.

        Please note that this differs from the `.pop` method of
        the `list` class  because it does not return the popped element.

        Returns
        -------
        optics: object
            The optical system with the layer removed.
        """
        new_layers: list = self.to_optics_list()
        _: object = new_layers.pop()
        dl_new_layers: dict = dl.utils.list_to_dictionary(new_layers)
        return eqx.tree_at(lambda x: x.layers, self, dl_new_layers)


class AlphaCentauri(dl.BinarySource):
    """
    A convinient representation of the Alpha Centauri binary system.

    Examples
    --------
    ```python
    >>> alpha_cen: object = AlphaCentauri()
    >>> wavelengths: float = 1e-09 * np.linspace(595., 695., 10)
    >>> fluxes: float = np.ones((10,), dtype = float)
    >>> spectrum: object = dl.ArraySpectrum(wavelengths, fluxes)
    >>> alpha_cen: object = AlphaCentauri(spectrum = spectrum)
    ```
    """

    def __init__(self: object, spectrum: float = None) -> object:
        """
        Simulate Alpha Centauri.

        Parameters
        ----------
        spectrum: float = None
            A `dl.Spectrum` if the default is not to be used. Recall
            that the convinience method `_simulate_alpha_cen_spectrum`
            can be used to simulate the spectrum.
        """
        if not spectrum:
            _spectrum: float = _read_csv_to_jax_array(SPECTRUM_DIR) 

            alpha_cen_a_waves: float = _spectrum[:, 0]
            alpha_cen_b_waves: float = _spectrum[:, 2]
            alpha_cen_a_flux: float = _spectrum[:, 1]
            alpha_cen_b_flux: float = _spectrum[:, 3]

            alpha_cen_waves: float = np.stack([alpha_cen_a_waves, alpha_cen_b_waves])
            alpha_cen_flux: float = np.stack([alpha_cen_a_flux, alpha_cen_b_flux])

            spectrum: float = dl.CombinedSpectrum(
                wavelengths=alpha_cen_waves, weights=alpha_cen_flux
            )

        super().__init__(
            position=ALPHA_CENTAURI_POSITION,
            flux=ALPHA_CENTAURI_MEAN_FLUX,
            contrast=ALPHA_CENTAURI_CONTRAST,
            separation=ALPHA_CENTAURI_SEPARATION,
            position_angle=ALPHA_CENTAURI_POSITION_ANGLE,
            spectrum=spectrum,
        )


class Background(dl.MultiPointSource):
    """
    Simplies the creation of a sample of background stars.

    The sample of background stars is pulled from the Gaia database
    but there is some voodoo involved in regularising the data.
    Use the `_simulate_background_stars` function to generate
    alternative samples.

    Examples
    --------
    ```python
    >>> bg: object = Background()
    >>> lim_bg: object = Background(number_of_bg_stars = 10)
    ```
    """

    def __init__(
        self: object, number_of_bg_stars: int = None, spectrum: object = None
    ) -> object:
        """
        Simulate background stars.

        Parameters
        ----------
        number_of_bg_stars: int = None
            How many background stars should be simulated.
        spectrum: object = None
            A `dl.Spectrum` if the default spectrum is not to be used.
        """
        if not spectrum:
            spectrum: object = dl.ArraySpectrum(
                wavelengths=np.linspace(
                    FILTER_MIN_WAVELENGTH, FILTER_MAX_WAVELENGTH, FILTER_DEFAULT_RES
                ),
                weights=np.ones((FILTER_DEFAULT_RES,), dtype=float),
            )

        # TODO: Better error handling if BACKGROUND_DIR is not valid
        _background: float = _read_csv_to_jax_array(BACKGROUND_DIR)

        if number_of_bg_stars:
            _background: float = _background[:number_of_bg_stars]

        position: float = _background[:, (0, 1)]
        flux: float = _background[:, 2]

        super().__init__(position=position, flux=flux, spectrum=spectrum)
