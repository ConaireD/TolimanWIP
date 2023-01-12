import jax.numpy as np
import jax
import equinox as eqx
import dLux as dl
import functools


def _downsample(arr: float, m: int) -> float:
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


def pixel_response(shape: float, threshold: float, seed: int = 1) -> float:
    key: object = jax.random.PRNGKey(seed)
    return 1. + threshold * jax.random.normal(key, shape)

DEFUALT_PUPIL_NPIX: int = 256
DEFUALT_DETECTOR_NPIX: int = 128
DEFUALT_NUMBER_OF_ZERNIKES: int = 5

TOLIMAN_PRIMARY_APERTURE_DIAMETER: float = .13
TOLIMAN_SECONDARY_MIRROR_DIAMETER: float = .032
TOLIMAN_DETECTOR_PIXEL_SIZE: float = .375
TOLIMAN_WIDTH_OF_STRUTS: float = .01
TOLIMAN_NUMBER_OF_STRUTS: int = 3

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

# TODO: I need to work out how to do the regularisation internally so 
#       that the values which are returned are always correct. 
# TODO: I need to make it so that the user can add and subtract layers
#       as they wish. 
class Toliman(dl.Optics):
    """
    """


    def __init__(
            self: object,
            simulate_polish: bool = True,
            simulate_aberrations: bool = True,
            simulate_jitter: bool = True,
            simulate_pixel_response: bool = True,
            simulate_
            operate_in_fresnel_mode: bool = False,
            operate_in_static_mode: bool = True,
            number_of_zernikes: int = DEFUALT_NUMBER_OF_ZERNIKES,
            pixels_in_pupil: int = DEFUALT_PUPIL_NPIX,
            pixels_on_detector: int = DEFUALT_DETECTOR_NPIX,
            path_to_mask: str = "assets/mask.npy",
            path_to_filter: str = "assets/filter.npy",
            path_to_polish: str = "assets/polish.npy") -> object:
        """
        """
        toliman_layers: list = [
            dl.CreateWavefront(
                pixels_in_pupil, 
                TOLIMAN_PRIMARY_APERTURE_DIAMETER,
                wavefront_type = "Angular"
            )
        ]

        # Adding the pupil
        dyn_toliman_pupil: object = dl.CompoundAperture(
            [
                dl.UniformSpider(
                    TOLIMAN_NUMBER_OF_STRUTS, 
                    TOLIMAN_WIDTH_OF_STRUTS
                ),
                dl.AnnularAperture(
                    TOLIMAN_PRIMARY_APERTURE_DIAMETER / 2., 
                    TOLIMAN_SECONDARY_MIRROR_DIAMETER / 2.
                )
            ]
        )

        if operate_in_static_mode:
            static_toliman_pupil: object = dl.StaticAperture(
                dyn_toliman_pupil,
                npixels = pixels_in_pupil,
                pixel_scale = TOLIMAN_PRIMARY_APERTURE_DIAMETER / pixels_in_pupil
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
                     dl.CircularAperture(
                         TOLIMAN_PRIMARY_APERTURE_DIAMETER / 2.
                     )
                ),
                coefficients = coeffs,
                npixels = pixels_in_pupil,
                pixel_scale = TOLIMAN_PRIMARY_APERTURE_DIAMETER / pixels_in_pupil
            )

            toliman_layers.append(toliman_aberrations)

        toliman_layers.append(dl.NormaliseWavefront())

        # Adding the propagator
        toliman_body: object
        if not operate_in_fresnel_mode:
            toliman_body: object = dl.AngularMFT(
                detector_npix,
                dl.utils.arcseconds_to_radians(pixels_in_detector)
            )
        else:
            raise NotImplementedError(FRESNEL_USE_ERR_MSG)

        toliman_layers.append(toliman_body)

        # Renormalising the flux.
        toliman_layers.append(dl.NormaliseWavefront())

        super().__init__(layers = toliman_layers)

        # Creating the default detector.
        toliman_jitter: object = dl.ApplyJitter(2.)
        toliman_saturation: object = dl.ApplySaturation(2500.)
            
        toliman_pixel_response: object = dl.ApplyPixelResponse(
            pixel_response((detector_npix, detector_npix), .05)
        )

        toliman_detector: object = dl.Detector(
            [toliman_pixel_response, toliman_jitter, toliman_saturation]
        )


        super().__init__(
            optics = toliman,
        )



