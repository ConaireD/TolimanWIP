import jax.numpy as np
import jax
import equinox as eqx
import dLux as dl
import functools

jax.config.update("jax_enable_x64", True)

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

DEFAULT_PUPIL_NPIX: int = 256
DEFAULT_DETECTOR_NPIX: int = 128
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
            pixels_in_pupil: int = DEFAULT_PUPIL_NPIX,
            pixels_on_detector: int = DEFAULT_DETECTOR_NPIX,
            path_to_mask: str = "assets/mask.npy",
            path_to_filter: str = "assets/filter.npy") -> object:
        """
        """
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

        shape: int = 5
        nolls: list = np.arange(2, shape + 2, dtype=int)
        true_coeffs: list = 1e-08 * jax.random.normal(jax.random.PRNGKey(0), (shape,))

        wavefront_factory: object = dl.CreateWavefront(
            npix, 
            aperture_diameter,
            wavefront_type = "Angular"
        )

        toliman_pupil: object = dl.StaticAperture(
            dl.CompoundAperture(
            [
                    dl.UniformSpider(
                        number_of_struts, 
                        width_of_struts
                    ),
                    dl.AnnularAperture(
                        aperture_diameter / 2., 
                        secondary_mirror_diameter / 2.
                    )
                ]
            ),
            npixels = npix,
            pixel_scale = aperture_diameter / npix
        )

        toliman_aberrations: object = dl.StaticAberratedAperture(
            dl.AberratedAperture(
                 nolls, 
                 true_coeffs, 
                 dl.CircularAperture(
                     aperture_diameter / 2.
                 )
            ),
            coefficients = true_coeffs,
            npixels = npix,
            pixel_scale = aperture_diameter / npix
        )

        toliman_mask: object = dl.AddOPD(mask)
        normalise: object = dl.NormaliseWavefront()

        toliman_body: object = dl.AngularMFT(
            detector_npix,
            dl.utils.arcseconds_to_radians(detector_pixel_size)
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

        def pixel_response(shape: float, threshold: float, seed: int = 1) -> float:
            key: object = jax.random.PRNGKey(seed)
            return 1. + threshold * jax.random.normal(key, shape)

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



