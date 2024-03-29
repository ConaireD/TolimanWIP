"""md
## Overview 
This file, amongst everything else contains the core optics of the `toliman` 
forwards model. The basic concept is that each component of the model is 
a specialisation (this is a type of inheritance) of a more general component.
Deciding what level of detail to include was very arbitrary. In some senses,
jut adding the pupil and the mask to the optics made the most sense as it 
gave the user the freedom to add other layers as they saw fit. But, in the 
end I decided that the simplest API was probably the best and so tried to 
include as much detail as possible. This is something that might change as
the model is used. The current model of the detector is a placeholder, since 
the actual physical detector has not been selected. As a result we cannot 
work with real values yet or include too much detail. 

There is a lot that can still be done with the forwards model, such as Fresnel
calculations, secondary mirror polish and sidelobes. Unfortunately, I never
got a chance to look into the sidelobes although it was high on my priority 
list. The problem is that acurrately simulating the diffraction grating 
requires very high sampling. In order to reduce the computation cost it was 
proposed that we avoid as many layers as possible when simulating the sidelobes
including only the aberrations. This then leads to complexity when considering 
how to make sure that the aberrations are shared between the models. 

## API
??? note "`TolimanDetector`"
    ::: toliman.toliman.TolimanDetector

??? note "`TolimanOptics`"
    ::: toliman.toliman.TolimanOptics

??? note "`AlphaCentauri`"
    ::: toliman.toliman.AlphaCentauri

??? note "`Background`"
    ::: toliman.toliman.Background
"""

import jax.numpy as np
import jax
import dLux as dl
import equinox as eqx
import os
import toliman.io as io
import toliman.constants as const # Runs on import
import toliman.collections as collections
import toliman.math as math

__author__ = "Jordan Dennis"
__all__ = [
    "TolimanDetector",
    "TolimanOptics",
    "AlphaCentauri",
    "Background",
]

class TolimanOptics(dl.Optics, collections.CollectionInterface):
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
        number_of_zernikes: int = int(os.environ["DEFAULT_NUMBER_OF_ZERNIKES"]),
        pixels_in_pupil: int = int(os.environ["DEFAULT_PUPIL_NPIX"]),
        pixels_on_detector: int = int(os.environ["DEFAULT_DETECTOR_NPIX"]),
        path_to_mask: str = os.environ["DEFAULT_MASK_DIR"],
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
        PRIMARY_APERTURE_DIAMETER: float = const.get_const_as_type("TOLIMAN_PRIMARY_APERTURE_DIAMETER", float)
        NUMBER_OF_STRUTS: int = const.get_const_as_type("TOLIMAN_NUMBER_OF_STRUTS", int)
        WIDTH_OF_STRUTS: float = const.get_const_as_type("TOLIMAN_WIDTH_OF_STRUTS", float)
        SECONDARY_MIRROR_DIAMETER: float = const.get_const_as_type("TOLIMAN_SECONDARY_MIRROR_DIAMETER", float)
        DETECTOR_PIXEL_SIZE: int = const.get_const_as_type("TOLIMAN_DETECTOR_PIXEL_SIZE", float)

        toliman_layers: list = [
            dl.CreateWavefront(
                pixels_in_pupil,
                PRIMARY_APERTURE_DIAMETER,
                wavefront_type="Angular",
            )
        ]

        # Adding the pupil
        dyn_toliman_pupil: object = dl.CompoundAperture(
            [
                dl.UniformSpider(NUMBER_OF_STRUTS, WIDTH_OF_STRUTS),
                dl.AnnularAperture(
                    PRIMARY_APERTURE_DIAMETER / 2.0,
                    SECONDARY_MIRROR_DIAMETER / 2.0,
                ),
            ]
        )

        if operate_in_static_mode:
            static_toliman_pupil: object = dl.StaticAperture(
                dyn_toliman_pupil,
                npixels=pixels_in_pupil,
                diameter=PRIMARY_APERTURE_DIAMETER,
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
                    raise NotImplementedError( 
                        "The mask you have loaded had a higher resolution " +\
                        "than the pupil. A method of resolving this has not " +\
                        "yet been created. Either change the value of the " +\
                        "`DEFAULT_PUPIL_NPIX` constant or use a different mask."
                    )
                if loaded_width % pixels_in_pupil == 0:
                    downsample_by: int = loaded_width // pixels_in_pupil
                    mask: float = math.downsample_square_grid(loaded_mask, downsample_by)
                else:
                    raise ValueError(
                        "The mask you have loaded could not be downsampled " +\
                        "onto the pupil. The shape of the mask was " +\
                        "({%i, %i}). Either change the value of the "
                        "environment variable `DEFAULT_PUPIL_NPIX` or use " +\
                        "a different mask.".format(loaded_width, loaded_width)
                    )
            else:
                mask: float = loaded_mask

            del loaded_mask
            del loaded_shape
            del loaded_width
        except IOError as ioe:
            raise ValueError(
                "The file address that of the mask did not exist. " +\
                "Make sure that you have a `.npy` representation of the " +\
                "phase mask available and have provided the correct file " +\
                "address to the constructor."
            )

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
                        PRIMARY_APERTURE_DIAMETER / 2.0
                    ),
                ),
                npixels=pixels_in_pupil,
                diameter=PRIMARY_APERTURE_DIAMETER,
            )

            toliman_layers.append(toliman_aberrations)

        toliman_layers.append(dl.NormaliseWavefront())

        if simulate_polish:
            raise NotImplementedError(
                "You have requested that the mirror polish be simulated " +\
                "this has not yet been implemented although it is planned " +\
                "in an upcoming release."
            )

        # Adding the propagator
        toliman_body: object
        if not operate_in_fresnel_mode:
            toliman_body: object = dl.AngularMFT(
                pixels_on_detector, DETECTOR_PIXEL_SIZE
            )
        else:
            raise NotImplementedError(
                "You have request operation in Fresenl mode. This has " +\
                "not currently been implemented. Once implemented it " +\
                "is not recommended that you use the feature as it is " +\
                "very slow and the zernike terms should be sufficient for " +\
                "most purposes."
            )

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


class TolimanDetector(dl.Detector, collections.CollectionInterface):
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
        DETECTOR_REPEATED_ERR_MSG = """
        You have provided a layer that is also a default layer. Make sure 
        that each type of detector is only provided once. 
        """

        DEFAULT_DETECTOR_JITTER: float = const.get_const_as_type("DEFAULT_DETECTOR_JITTER", float)
        DEFAULT_DETECTOR_SATURATION: float = const.get_const_as_type("DEFAULT_DETECTOR_SATURATION", float)
        DEFAULT_DETECTOR_NPIX: int = const.get_const_as_type("DEFAULT_DETECTOR_NPIX", int)
        DEFAULT_DETECTOR_THRESHOLD: float = const.get_const_as_type("DEFAULT_DETECTOR_THRESHOLD", float)

        detector_layers: list = []

        if simulate_jitter:
            detector_layers.append(dl.ApplyJitter(DEFAULT_DETECTOR_JITTER))

            # TODO: Make a contains instance function
            if collections.contains_instance(extra_detector_layers, dl.ApplyJitter):
                raise ValueError(DETECTOR_REPEATED_ERR_MSG)

        if simulate_saturation:
            detector_layers.append(dl.ApplySaturation(DEFAULT_DETECTOR_SATURATION))

            if collections.contains_instance(extra_detector_layers, dl.ApplySaturation):
                raise ValueError(DETECTOR_REPEATED_ERR_MSG)

        if simulate_pixel_response:
            detector_layers.append(
                dl.ApplyPixelResponse(
                    math.pixel_response(
                        (DEFAULT_DETECTOR_NPIX, DEFAULT_DETECTOR_NPIX),
                        DEFAULT_DETECTOR_THRESHOLD,
                    )
                )
            )

            if collections.contains_instance(extra_detector_layers, dl.ApplyPixelResponse):
                raise ValueError(DETECTOR_REPEATED_ERR_MSG)

        detector_layers.extend(extra_detector_layers)

        if detector_layers:
            super().__init__(detector_layers)
        else:
            raise ValueError(
                "You have provided no detector layers and not asked " +\
                "for any of the defaults. This implies that the detector " +\
                "does not contain any layers which is not a valid state. " +\
                "If you do not wish to model any detector effects do not " +\
                "provide a detector in construction."
            )

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
        SPECTRUM_DIR: str = const.get_const_as_type("SPECTRUM_DIR", str) 

        if not spectrum:
            _spectrum: float = io.read_csv_to_jax_array(SPECTRUM_DIR) 

            alpha_cen_a_waves: float = _spectrum[:, 0]
            alpha_cen_b_waves: float = _spectrum[:, 2]
            alpha_cen_a_flux: float = _spectrum[:, 1]
            alpha_cen_b_flux: float = _spectrum[:, 3]

            alpha_cen_waves: float = np.stack([alpha_cen_a_waves, alpha_cen_b_waves])
            alpha_cen_flux: float = np.stack([alpha_cen_a_flux, alpha_cen_b_flux])

            spectrum: float = dl.CombinedSpectrum(
                wavelengths=alpha_cen_waves, weights=alpha_cen_flux
            )

        ALPHA_CENTAURI_POSITION: float = const.get_const_as_type("ALPHA_CENTAURI_POSITION", np.array)
        ALPHA_CENTAURI_MEAN_FLUX: float = const.get_const_as_type("ALPHA_CENTAURI_MEAN_FLUX", float)
        ALPHA_CENTAURI_CONTRAST: float = const.get_const_as_type("ALPHA_CENTAURI_CONTRAST", float)
        ALPHA_CENTAURI_SEPARATION: float = const.get_const_as_type("ALPHA_CENTAURI_SEPARATION", float)
        ALPHA_CENTAURI_POSITION_ANGLE: float = const.get_const_as_type("ALPHA_CENTAURI_POSITION_ANGLE", float)

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
        FILTER_MIN_WAVELENGTH: float = const.get_const_as_type("FILTER_MIN_WAVELENGTH", float)
        FILTER_MAX_WAVELENGTH: float = const.get_const_as_type("FILTER_MAX_WAVELENGTH", float)
        BG_DEF_SPEC_RES: int = const.get_const_as_type("BACKGROUND_STAR_SPEC_RES", int)
        BACKGROUND_DIR: str = const.get_const_as_type("BACKGROUND_DIR", str) 

        if not spectrum:
            spectrum: object = dl.ArraySpectrum(
                wavelengths=np.linspace(
                    FILTER_MIN_WAVELENGTH, FILTER_MAX_WAVELENGTH, BG_DEF_SPEC_RES
                ),
                weights=np.ones((BG_DEF_SPEC_RES,), dtype=float),
            )

        # TODO: Better error handling if BACKGROUND_DIR is not valid
        _background: float = io.read_csv_to_jax_array(BACKGROUND_DIR)

        if number_of_bg_stars:
            select_by: int = _background.shape[0] // number_of_bg_stars
            _background: float = _background[::select_by, :]

        position: float = np.pi / 180 * _background[:, (0, 1)]
        flux: float = 1e5 * _background[:, 2]

        super().__init__(position=position, flux=flux, spectrum=spectrum)
