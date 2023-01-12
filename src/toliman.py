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

class Toliman(dl.Instrument):
    """
    """


    def __init__(
            self: object,
            number_of_zernikes: int = DEFUALT_NUMBER_OF_ZERNIKES,
            pixels_in_pupil: int = DEFUALT_PUPIL_NPIX,
            pixels_on_detector: int = DEFUALT_DETECTOR_NPIX,
            path_to_mask: str = "assets/mask.npy",
            path_to_filter: str = "assets/filter.npy") -> object:
        """
        """
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

        # Generating the Zernikes
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

        wavefront_factory: object = dl.CreateWavefront(
            pixels_in_pupil, 
            TOLIMAN_PRIMARY_APERTURE_DIAMETER,
            wavefront_type = "Angular"
        )

        toliman_pupil: object = dl.StaticAperture(
            dl.CompoundAperture(
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
            ),
            npixels = pixels_in_pupil,
            pixel_scale = TOLIMAN_PRIMARY_APERTURE_DIAMETER / pixels_in_pupil
        )

        toliman_mask: object = dl.AddOPD(mask)
        normalise: object = dl.NormaliseWavefront()

        toliman_body: object = dl.AngularMFT(
            detector_npix,
            dl.utils.arcseconds_to_radians(pixels_in_detector)
        )

        toliman: object = dl.Optics(
            layers = [
                wavefront_factory,
                toliman_pupil,
                toliman_aberrations,
                toliman_mask,
                normalise,
                toliman_body,
                normalise
            ]
        )

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
            sources = [alpha_centauri],
            detector = toliman_detector
        )



