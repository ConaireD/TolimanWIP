import jax.numpy as np
import jax
import dLux as dl
import equinox as eqx


__author__ = "Jordan Dennis"
__all__ = ["TolimanDetector", "TolimanOptics", "AlphaCentauri", "Background", "_contains_instance"]


def _contains_instance(_list: list, _type: type) -> bool:
    """
    Check to see if a list constains an element of a certain
    type.

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
    Resample a square array by a factor of `m`. `

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


def _simulate_alpha_cen_spectra(number_of_wavelenths: int = 25) -> None:
    """
    This function will simulate the spectrum of the alpha centauri
    binary using `pysynphot`. The output is saved to a file so that
    it can be used again later without having to be reloaded.

    Parameters
    ----------
    number_of_wavelengts: int
        The number of wavelengths that you wish to use for the simulation.
        The are taken from the `pysynphot` output by binning.
    """
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

    m: int = alpha_cen_a_waves.size // number_of_wavelenths

    alpha_cen_a_waves: float = _downsample_along_axis(alpha_cen_a_spectrum.wave, m)

    alpha_cen_a_flux: float = _downsample_along_axis(alpha_cen_a_spectrum.flux, m)

    alpha_cen_b_waves: float = _downsample_along_axis(alpha_cen_b_spectrum.wave, m)

    alpha_cen_b_flux: float = _downsample_along_axis(alpha_cen_b_spectrum.flux, m)

    with open("assets/spectra.csv", "w") as spectra:
        spectra.write("alpha cen a waves (m), ")
        spectra.write("alpha cen a flux (W/m/m), ")
        spectra.write("alpha cen b waves (m), ")
        spectra.write("alpha cen b flux (W/m/m)\n")

        for i in np.arange(number_of_wavelenths, dtype=int):
            spectra.write("%f, ".format(alpha_cen_a_waves[i]))
            spectra.write("%f, ".format(alpha_cen_a_flux[i]))
            spectra.write("%f, ".format(alpha_cen_b_waves[i]))
            spectra.write("%f\n".format(alpha_cen_b_flux[i]))


# TODO: Add arguments
# TODO: optimise
def _simulate_background_stars() -> None:
    """
    This function samples the Gaia database for a typical sample
    of background stars. The primary use of this function is to
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

    ra_in_range: float = np.abs(bg_stars_ra - bg_ra) < bg_win
    dec_in_range: float = np.abs(bg_stars_dec - bg_dec) < bg_win
    in_range: float = ra_in_range & dec_in_range
    sample_len: float = in_range.sum()

    bg_stars_ra_crop: float = bg_stars_ra[in_range]
    bg_stars_dec_crop: float = bg_stars_dec[in_range]
    bg_stars_flux_crop: float = bg_stars_flux[in_range]
    bg_stars_rel_flux_crop: float = bg_stars_flux_crop / alpha_cen_flux

    with open("datasheets/bg_stars.csv", "w") as sheet:
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
    A convinience wrapper for generating a pixel reponse array.

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
    A convinience wrapper for generating photon noise.

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


DEFAULT_PUPIL_NPIX: int = 256
DEFAULT_DETECTOR_NPIX: int = 128
DEFAULT_NUMBER_OF_ZERNIKES: int = 5

TOLIMAN_PRIMARY_APERTURE_DIAMETER: float = 0.13
TOLIMAN_SECONDARY_MIRROR_DIAMETER: float = 0.032
TOLIMAN_DETECTOR_PIXEL_SIZE: float = 0.375
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
ALPHA_CEN_B_METALLICITY: float = 0.23
ALPHA_CEN_B_SURFACE_GRAV: float = 4.37

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


class ExtendableModule(eqx.Module):
    """
    This "interface" adds `.insert` and `.remove` methods to an
    `equinox` module. The interface is based of the interface of
    the default python list. This class is designed to be used in
    multiple inheritance with other classes and should not be used
    on it's own.
    """

    def to_optics_list(self: object) -> list:
        """
        Get the optical elements that make up the object as a list. 

        Returns
        -------
        optics: list
            The optical layers in order in a list.
        """
        return list(self.layers.values())


    # TODO: Add a remove method
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
        if not isinstance(optic, dl.OpticalLayer):
            raise ValueError("Inserted optics must be optical layers.")

        new_layers: list = self.layers.copy().insert(index, optic)
        return eqx.tree_at(lambda x: x.layers, self, new_layers)

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
        if not isinstance(optic, dl.OpticalLayer):
            raise ValueError("Inserted optics must be optical layers.")

        new_layers: list = self.layers.copy().remove(optic)
        return eqx.tree_at(lambda x: x.layers, self, new_layers)


# TODO: I need to work out how to do the regularisation internally so
#       that the values which are returned are always correct.
class TolimanOptics(dl.Optics, ExtendableModule):
    """
    Simulates the optical system of the TOLIMAN telescope. It is
    designed to occupy the `optics` kwarg of `dl.Instrument`.
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
        simulate_polish: bool = True,
        simulate_aberrations: bool = True,
        operate_in_fresnel_mode: bool = False,
        operate_in_static_mode: bool = True,
        number_of_zernikes: int = DEFAULT_NUMBER_OF_ZERNIKES,
        pixels_in_pupil: int = DEFAULT_PUPIL_NPIX,
        pixels_on_detector: int = DEFAULT_DETECTOR_NPIX,
        path_to_mask: str = "assets/mask.npy",
        path_to_filter: str = "assets/filter.npy",
        path_to_polish: str = "assets/polish.npy",
    ) -> object:
        """
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
                pixel_scale=TOLIMAN_PRIMARY_APERTURE_DIAMETER / pixels_in_pupil,
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
                    mask: float = _downsample(loaded_mask, downsample_by)
                else:
                    raise ValueError(MASK_SAMPLING_ERR_MSG)
            else:
                mask: float = loaded_mask

            del loaded_mask
            del loaded_shape
            del loaded_width
        except IOError as ioe:
            raise ValueError(MASK_IMPORT_ERR_MSG)

        toliman_model.append(dl.AddOPD(mask))

        # Generating the Zernikes
        # TODO: Make zernike_coefficients a function
        if simulate_aberrations:
            nolls: list = np.arange(2, number_of_zernikes + 2, dtype=int)
            seed: object = jax.random.PRNGKey(0)
            coeffs: list = 1e-08 * jax.random.normal(seed, (number_of_zernikes,))

            toliman_aberrations: object = dl.StaticAberratedAperture(
                dl.AberratedAperture(
                    nolls,
                    coeffs,
                    dl.CircularAperture(TOLIMAN_PRIMARY_APERTURE_DIAMETER / 2.0),
                ),
                coefficients=coeffs,
                npixels=pixels_in_pupil,
                pixel_scale=TOLIMAN_PRIMARY_APERTURE_DIAMETER / pixels_in_pupil,
            )

            toliman_layers.append(toliman_aberrations)

        toliman_layers.append(dl.NormaliseWavefront())

        if simulate_polish:
            raise NotImplementedError(POLISH_USE_ERR_MSG)

        # Adding the propagator
        toliman_body: object
        if not operate_in_fresnel_mode:
            toliman_body: object = dl.AngularMFT(
                detector_npix, dl.utils.arcseconds_to_radians(pixels_in_detector)
            )
        else:
            raise NotImplementedError(FRESNEL_USE_ERR_MSG)

        toliman_layers.append(toliman_body)

        # Renormalising the flux.
        toliman_layers.append(dl.NormaliseWavefront())

        super().__init__(layers=toliman_layers)


class TolimanDetector(dl.Detector, ExtendableModule):
    """
    A default implementation of a generic detector that is designed
    to be used with the `dLux.Instrument`.

    Examples
    --------
    ```py
    >>> toliman_detector: object = TolimanDetector()
    >>> toliman_detector: object = TolimanDetector(simulate_jitter = False)
    """

    def __init__(
        self: object,
        simulate_jitter: bool = True,
        simulate_pixel_response: bool = True,
        simulate_saturation: bool = True,
        extra_detector_layers: list = [],
    ) -> object:
        """
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

            if _contains_instance(detector_layers, dl.ApplySaturation):
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

            if _contains_instance(detector_layers, dl.ApplyPixelResponse):
                raise ValueError(DETECTOR_REPEATED_ERR_MSG)

        detector_layers.extend(extra_detector_layers)

        if detector_layers:
            super().__init__(detector_layers)
        else:
            raise ValueError(DETECTOR_EMPTY_ERR_MSG)


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
    """

    def __init__(self: object, spectrum: float = None) -> object:
        """
        Parameters
        ----------
        spectrum: float = None
            A `dl.Spectrum` if the default is not to be used. Recall
            that the convinience method `_simulate_alpha_cen_spectrum`
            can be used to simulate the spectrum.
        """
        if not spectrum:
            with open(SPECTRUM_DIR, "r") as spectrum:
                lines: list = open.readlines().remove(0)

                strip: callable = lambda _str: _str.strip().split(",")
                str_to_float: callable = lambda _str: float(_str.strip())
                entries: list = jax.tree_map(strip, lines)
                _spectrum: float = jax.tree_map(str_to_float, entries)

            alpha_cen_waves: float = _spectrum[:, (0, 2)]
            alpha_cen_flux: float = _spectrum[:, (1, 3)]

            spectrum: float = dl.CombinedSpectrum(
                wavelengths=alpha_cen_waves, flux=alpha_cen_flux
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
    Simplies the creation of a sample of background stars. The
    sample of background stars is pulled from the Gaia database
    but there is some voodoo involved in regularising the data.
    Use the `_simulate_background_stars` function to generate
    alternative samples.

    Examples
    --------
    >>> bg: object = Background()
    >>> lim_bg: object = Background(number_of_bg_stars = 10)
    """

    def __init__(
        self: object, number_of_bg_stars: int = None, spectrum: object = None
    ) -> object:
        """
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
                fluxes=np.ones((FILTER_DEFAULT_RES,), dtype=float),
            )

        # TODO: Better error handling if BACKGROUND_DIR is not valid
        with open(BACKGROUND_DIR, "r") as background:
            lines: list = background.readlines().remove(0)
            strip: callable = lambda _str: _str.strip().split(",")
            str_to_float: callable = lambda _str: float(_str.strip())
            entries: list = jax.tree_map(strip, lines)
            _background: float = jax.tree_map(str_to_float, entries)

            if number_of_bg_stars:
                _background: float = _background[:number_of_stars]

        position: float = _background[:, (0, 1)]
        flux: float = _background[:, 2]

        super().__init__(position=position, flux=flux, spectrum=spectrum)