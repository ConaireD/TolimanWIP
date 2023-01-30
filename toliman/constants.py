import os
import dLux as dl

os.environ["DEFAULT_PUPIL_NPIX"]: int = "256"
os.environ["DEFAULT_DETECTOR_NPIX"]: int = "128"
os.environ["DEFAULT_NUMBER_OF_ZERNIKES"]: int = "5"

os.environ["DEFAULT_MASK_DIR"]: str = "/home/jordan/Documents/toliman/toliman/assets/mask.npy"
os.environ["SPECTRUM_DIR"]: str = "/home/jordan/Documents/toliman/toliman/assets/spectra.csv"
os.environ["BACKGROUND_DIR"]: str = "toliman/assets/background.csv"

os.environ["TOLIMAN_PRIMARY_APERTURE_DIAMETER"]: float = "0.13"
os.environ["TOLIMAN_SECONDARY_MIRROR_DIAMETER"]: float = "0.032"
os.environ["TOLIMAN_DETECTOR_PIXEL_SIZE"]: float = "dl.utils.arcseconds_to_radians(0.375)"
os.environ["TOLIMAN_WIDTH_OF_STRUTS"]: float = "0.01"
os.environ["TOLIMAN_NUMBER_OF_STRUTS"]: int = "3"

os.environ["DEFAULT_DETECTOR_JITTER"]: float = "2.0"
os.environ["DEFAULT_DETECTOR_SATURATION"]: float = "2500"
os.environ["DEFAULT_DETECTOR_THRESHOLD"]: float = "0.05"

os.environ["ALPHA_CENTAURI_SEPARATION"]: float = "dl.utils.arcseconds_to_radians(8.0)"
os.environ["ALPHA_CENTAURI_POSITION"]: float = "np.array([0.0, 0.0], dtype=float)"
os.environ["ALPHA_CENTAURI_MEAN_FLUX"]: float = "1.0"
os.environ["ALPHA_CENTAURI_CONTRAST"]: float = "2.0"
os.environ["ALPHA_CENTAURI_POSITION_ANGLE"]: float = "0.0"

os.environ["ALPHA_CEN_A_SURFACE_TEMP"]: float = "5790.0"
os.environ["ALPHA_CEN_A_METALICITY"]: float = "0.2"
os.environ["ALPHA_CEN_A_SURFACE_GRAV"]: float = "4.0"

os.environ["ALPHA_CEN_B_SURFACE_TEMP"]: float = 5260.0
os.environ["ALPHA_CEN_B_METALICITY"]: float = 0.23
os.environ["ALPHA_CEN_B_SURFACE_GRAV"]: float = 4.37

os.environ["FILTER_MIN_WAVELENGTH"]: float = 595e-09
os.environ["FILTER_MAX_WAVELENGTH"]: float = 695e-09
os.environ["FILTER_DEFAULT_RES"]: int = 24


