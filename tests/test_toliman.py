import pytest
import os

from dLux import (
    CircularAperture,
    HexagonalAperture,
    AddOPD,
    ApplyJitter,
    ApplyPixelResponse,
    StaticAperture,
    StaticAberratedAperture,
    ArraySpectrum,
    CombinedSpectrum,
    CreateWavefront,
    ApplyJitter,
    ApplySaturation,
    ApplyPixelResponse,
    AddConstant,
)

from toliman import (
    TolimanOptics,
    TolimanDetector,
    AlphaCentauri,
    Background,
)

from toliman.collections import contains_instance
from toliman.io import read_csv_to_jax_array
import toliman.constants as const

from jax import (
    numpy as np,
    jit, 
    grad, 
    vmap
)


class GeometricAberrations(object):
    """
    """
    pass


class FresnelPropagator(object):
    """
    """
    pass

MASK_DIR: str = const.get_const_as_type("DEFAULT_MASK_DIR", str)

class TestTolimanOptics(object):
    @pytest.mark.xdist_group("2")
    @pytest.mark.software
    def test_constructor_when_static(self: object) -> None:
        # Arrange/Act
        static_toliman: object = TolimanOptics(operate_in_static_mode=True)

        # Assert
        optics: list = static_toliman.to_optics_list()
        assert contains_instance(optics, StaticAperture)

    @pytest.mark.xdist_group("1")
    @pytest.mark.software
    def test_constructor_when_not_static(self: object) -> None:
        # Arrange/Act
        dynamic_toliman: object = TolimanOptics(operate_in_static_mode=False)

        # Assert
        optics: list = dynamic_toliman.to_optics_list()
        assert not contains_instance(optics, StaticAperture)

    @pytest.mark.xdist_group("2")
    @pytest.mark.software
    def test_constructor_when_mask_too_large(self: object) -> None:
        with pytest.raises(NotImplementedError):
            toliman: object = TolimanOptics(pixels_in_pupil=2048)

    @pytest.mark.xdist_group("3")
    @pytest.mark.software
    def test_constructor_when_mask_incorrectly_sampled(self: object) -> None:
        # Arrange/Act/Assert
        with pytest.raises(ValueError):
            toliman: object = TolimanOptics(pixels_in_pupil=125)

    @pytest.mark.xdist_group("1")
    @pytest.mark.software
    def test_constructor_when_mask_is_correct(self: object) -> None:
        # Arrange/Act
        toliman: object = TolimanOptics(pixels_in_pupil=256)

        # Assert
        optics: list = toliman.to_optics_list()
        assert contains_instance(optics, AddOPD)

    @pytest.mark.xdist_group("4")
    @pytest.mark.software
    @pytest.mark.skipif(not os.path.isfile(MASK_DIR), reason="The mask is not installed.")
    def test_constructor_when_mask_is_correct_at_max(self: object) -> None:
        # TODO: Load the mask to get the correct size
        mask: float = np.load(MASK_DIR)
        max_size: int = mask.shape[0]
        # Arrange/Act
        toliman: object = TolimanOptics(pixels_in_pupil=max_size)

        # Assert
        optics: list = toliman.to_optics_list()
        assert contains_instance(optics, AddOPD)

    @pytest.mark.xdist_group("1")
    @pytest.mark.software
    def test_constructor_when_mask_file_is_incorrect(self: object) -> None:
        # Incorrect file address error.
        # TODO: Make sure that this file does not exist
        #       This is a best practice thing as it can theoretically
        #       exist making this test environment dependent. This
        #       violates the R in the F.I.R.S.T principles.
        with pytest.raises(ValueError):
            toliman: object = TolimanOptics(path_to_mask="i/don't/exist.npy")

    @pytest.mark.xdist_group("4")
    @pytest.mark.software
    def test_constructor_when_aberrated(self: object) -> None:
        # Arrange/Act
        toliman: object = TolimanOptics(simulate_aberrations=True)
        optics: list = toliman.to_optics_list()

        # Assert
        assert contains_instance(optics, StaticAberratedAperture)

    @pytest.mark.xdist_group("2")
    @pytest.mark.software
    def test_constructor_when_not_aberrated(self: object) -> None:
        # Arrange/Act
        toliman: object = TolimanOptics(simulate_aberrations=False)
        optics: list = list(toliman.layers.values())

        # Assert
        assert not contains_instance(optics, StaticAberratedAperture)

    @pytest.mark.xdist_group("3")
    @pytest.mark.software
    def test_constructor_when_polish_is_simulated(self: object) -> None:
        # Arrange/Act/Assert
        with pytest.raises(NotImplementedError):
            toliman: object = TolimanOptics(simulate_polish=True)

    @pytest.mark.xdist_group("4")
    @pytest.mark.software
    def test_constructor_when_polish_is_not_simulated(self: object) -> None:
        # Arrange/Act
        toliman: object = TolimanOptics(simulate_polish=False)
        optics: list = toliman.to_optics_list()

        # Assert
        assert not contains_instance(optics, GeometricAberrations)

    @pytest.mark.xdist_group("3")
    @pytest.mark.software
    def test_constructor_when_using_fresnel(self: object) -> None:
        # Operate in Fresnel mode
        with pytest.raises(NotImplementedError):
            toliman: object = TolimanOptics(operate_in_fresnel_mode=True)

    @pytest.mark.xdist_group("1")
    @pytest.mark.software
    def test_constructor_when_not_using_fresnel(self: object) -> None:
        # Arrange/Act
        toliman: object = TolimanOptics(operate_in_fresnel_mode=False)
        optics: list = toliman.to_optics_list()

        # TODO: get the correct name for the FresnelPropagator.
        # Assert
        assert not contains_instance(optics, FresnelPropagator)

    @pytest.mark.xdist_group("4")
    @pytest.mark.software
    def test_insert_when_type_is_incorrect(self: object) -> None:
        # Arrange
        toliman: object = TolimanOptics()
        with pytest.raises(ValueError):
            # Act/Assert
            toliman.insert(0, 1)

    @pytest.mark.xdist_group("1")
    @pytest.mark.software
    def test_insert_when_index_is_too_long(self: object) -> None:
        # Arrange
        toliman: object = TolimanOptics()
        optics: list = toliman.to_optics_list()
        length: int = len(optics)
        wrong_index: int = length + 1
        element_to_insert: object = CircularAperture(1.0)

        # Act/Assert
        with pytest.raises(ValueError):
            toliman.insert(wrong_index, element_to_insert)

    @pytest.mark.xdist_group("2")
    @pytest.mark.software
    def test_insert_when_index_is_negative(self: object) -> None:
        # Arrange
        toliman: object = TolimanOptics()
        element_to_insert: object = CircularAperture(1.0)

        # Act/Assert
        with pytest.raises(ValueError):
            toliman.insert(-1, element_to_insert)

    @pytest.mark.xdist_group("3")
    @pytest.mark.software
    def test_insert_when_correct(self: object) -> None:
        # Arrange
        toliman: object = TolimanOptics()
        optics: list = toliman.to_optics_list()
        length: int = len(optics)

        # Act
        insertion: object = HexagonalAperture(1.0)
        toliman: object = toliman.insert(insertion, 0)

        # Assert
        new_optics: list = toliman.to_optics_list()
        assert contains_instance(new_optics, HexagonalAperture)

    @pytest.mark.xdist_group("4")
    @pytest.mark.software
    def test_remove_when_index_is_too_long(self: object) -> None:
        # Arrange
        toliman: object = TolimanOptics()
        optics: list = toliman.to_optics_list()
        length: int = len(optics)
        wrong_index: int = length + 1

        # Act/Assert
        with pytest.raises(ValueError):
            toliman.remove(wrong_index)

    @pytest.mark.xdist_group("1")
    @pytest.mark.software
    def test_remove_when_index_is_negative(self: object) -> None:
        # Arrange
        toliman: object = TolimanOptics()

        # Act/Assert
        with pytest.raises(ValueError):
            toliman.remove(-1)

    @pytest.mark.xdist_group("2")
    @pytest.mark.software
    def test_remove_when_correct(self: object) -> None:
        # Arrange
        toliman: object = TolimanOptics()
        optics: list = toliman.to_optics_list()

        # Act
        new_toliman: object = toliman.remove(0)
        new_optics: list = new_toliman.to_optics_list()

        # Assert
        assert not contains_instance(new_optics, CreateWavefront)

    @pytest.mark.xdist_group("3")
    @pytest.mark.software
    def test_append_when_type_is_incorrect(self: object) -> None:
        # Arrange
        toliman: object = TolimanOptics()

        # Act/Assert
        with pytest.raises(ValueError):
            toliman.append(1)

    @pytest.mark.xdist_group("4")
    @pytest.mark.software
    def test_append_when_correct(self: object) -> None:
        # Arrange
        toliman: object = TolimanOptics()
        element_to_append: object = CircularAperture(1.0)

        # Act
        toliman: object = toliman.append(element_to_append)
        optics: list = toliman.to_optics_list()

        # Assert
        assert contains_instance(optics, CircularAperture)

    @pytest.mark.xdist_group("1")
    @pytest.mark.software
    def test_pop_removes_element(self: object) -> None:
        # Arrange
        toliman: object = TolimanOptics()
        optics: list = toliman.to_optics_list()

        # Act
        new_toliman: object = toliman.pop()
        new_optics: list = new_toliman.to_optics_list()

        # Assert
        assert optics[-1] != new_optics[-1]

#    @pytest.mark.physical
#    def test_psf_central(self: object) -> None:
#
#    @pytest.mark.physical
#    def test_mask_correct_order_of_magnitude(self: object) -> None:


class TestTolimanDetector(object):
    @pytest.mark.xdist_group("2")
    @pytest.mark.software
    def test_constructor_when_jittered(self: object) -> None:
        # Arrange/Act
        detector: object = TolimanDetector(simulate_jitter=True)
        optics: list = detector.to_optics_list()

        # Assert
        assert contains_instance(optics, ApplyJitter)

    @pytest.mark.xdist_group("3")
    @pytest.mark.software
    def test_constructor_when_not_jittered(self: object) -> None:
        # Arrange/Act
        detector: object = TolimanDetector(simulate_jitter=False)
        optics: list = detector.to_optics_list()

        # Assert
        assert not contains_instance(optics, ApplyJitter)

    @pytest.mark.xdist_group("4")
    @pytest.mark.software
    def test_constructor_when_jitter_is_repeated(self: object) -> None:
        # Arrange
        jitter: object = ApplyJitter(2.0)

        # Act/Assert
        with pytest.raises(ValueError):
            detector: object = TolimanDetector(
                simulate_jitter=True, extra_detector_layers=[jitter]
            )

    @pytest.mark.xdist_group("1")
    @pytest.mark.software
    def test_constructor_when_saturated(self: object) -> None:
        # Arrange/Act
        detector: object = TolimanDetector(simulate_saturation=True)
        optics: list = detector.to_optics_list()

        # Assert
        assert contains_instance(optics, ApplySaturation)

    @pytest.mark.xdist_group("2")
    @pytest.mark.software
    def test_constructor_when_not_saturated(self: object) -> None:
        # Arrange/Act
        detector: object = TolimanDetector(simulate_saturation=False)
        optics: list = detector.to_optics_list()

        # Assert
        assert not contains_instance(optics, ApplySaturation)

    @pytest.mark.xdist_group("3")
    @pytest.mark.software
    def test_constructor_when_saturation_is_repeated(self: object) -> None:
        # Arrange
        saturation: object = ApplySaturation(2)

        # Act/Assert
        with pytest.raises(ValueError):
            detector: object = TolimanDetector(
                simulate_saturation=True, extra_detector_layers=[saturation]
            )

    @pytest.mark.xdist_group("4")
    @pytest.mark.software
    def test_constructor_when_pixels_respond(self: object) -> None:
        # Arrange/Act
        detector: object = TolimanDetector(simulate_pixel_response=True)
        optics: list = detector.to_optics_list()

        # Assert
        assert contains_instance(optics, ApplyPixelResponse)

    @pytest.mark.xdist_group("1")
    @pytest.mark.software
    def test_constructor_when_pixels_dont_respond(self: object) -> None:
        # Arrange/Act
        detector: object = TolimanDetector(simulate_pixel_response=False)
        optics: list = detector.to_optics_list()

        # Assert
        assert not contains_instance(optics, ApplyPixelResponse)

    @pytest.mark.xdist_group("2")
    @pytest.mark.software
    def test_constructor_when_pixel_response_is_repeated(self: object) -> None:
        # Arrange
        pixel_response: object = ApplySaturation(2)

        # Act/Assert
        with pytest.raises(ValueError):
            detector: object = TolimanDetector(
                simulate_pixel_response=True, extra_detector_layers=[pixel_response]
            )

    @pytest.mark.xdist_group("3")
    @pytest.mark.software
    def test_constructor_when_correct(self: object) -> None:
        # Arrange/Act
        detector: object = TolimanDetector()
        optics: list = detector.to_optics_list()

        # Assert
        assert contains_instance(optics, ApplyJitter)
        assert contains_instance(optics, ApplySaturation)
        assert contains_instance(optics, ApplyPixelResponse)

    @pytest.mark.xdist_group("4")
    @pytest.mark.software
    def test_constructor_when_empty(self: object) -> None:
        # Arrange/Act/Assert
        with pytest.raises(ValueError):
            detector: object = TolimanDetector(
                simulate_jitter=False,
                simulate_saturation=False,
                simulate_pixel_response=False,
            )

    @pytest.mark.xdist_group("1")
    @pytest.mark.software
    def test_insert_when_type_is_incorrect(self: object) -> None:
        # Arrange
        detector: object = TolimanDetector()

        # Act/Assert
        with pytest.raises(ValueError):
            detector.insert(0, 1)

    @pytest.mark.xdist_group("2")
    @pytest.mark.software
    def test_insert_when_index_is_too_long(self: object) -> None:
        # Arrange
        detector: object = TolimanDetector()
        optics: object = detector.to_optics_list()
        length: int = len(optics)
        too_long: int = length + 1
        element_to_insert: object = AddConstant(1.0)

        # Act/Assert
        with pytest.raises(ValueError):
            detector.insert(too_long, element_to_insert)

    @pytest.mark.xdist_group("3")
    @pytest.mark.software
    def test_insert_when_index_is_negative(self: object) -> None:
        # Arrange
        detector: object = TolimanDetector()
        element_to_insert: object = AddConstant(1.0)

        # Act/Assert
        with pytest.raises(ValueError):
            detector.insert(-1, element_to_insert)

    @pytest.mark.xdist_group("4")
    @pytest.mark.software
    def test_insert_when_correct(self: object) -> None:
        # Arrange
        detector: object = TolimanDetector()
        element_to_insert: object = AddConstant(1.0)

        # Act
        detector: object = detector.insert(element_to_insert, 0)
        optics: list = detector.to_optics_list()

        # Assert
        assert contains_instance(optics, AddConstant)

    @pytest.mark.xdist_group("1")
    @pytest.mark.software
    def test_remove_when_index_is_too_long(self: object) -> None:
        # Arrange
        detector: object = TolimanDetector()
        optics: list = detector.to_optics_list()
        length: int = len(optics)
        too_long: int = length + 1

        # Act/Assert
        with pytest.raises(ValueError):
            detector.remove(too_long)

    @pytest.mark.xdist_group("2")
    @pytest.mark.software
    def test_remove_when_index_is_negative(self: object) -> None:
        # Arrange
        detector: object = TolimanDetector()

        # Act
        with pytest.raises(ValueError):
            detector.remove(-1)

    @pytest.mark.xdist_group("3")
    @pytest.mark.software
    def test_remove_when_correct(self: object) -> None:
        # Arrange
        detector: object = TolimanDetector()

        # Act
        detector: object = detector.remove(0)
        optics: list = detector.to_optics_list()

        # Assert
        assert not contains_instance(optics, ApplyJitter)

    @pytest.mark.xdist_group("4")
    @pytest.mark.software
    def test_append_when_type_is_incorrect(self: object) -> None:
        # Arrange
        detector: object = TolimanDetector()

        # Act/Assert
        with pytest.raises(ValueError):
            detector.append(1)

    @pytest.mark.xdist_group("1")
    @pytest.mark.software
    def test_append_when_correct(self: object) -> None:
        # Arrange
        detector: object = TolimanDetector()
        element_to_insert: object = AddConstant(2.0)

        # Act
        new_detector: object = detector.append(element_to_insert)
        new_optics: list = new_detector.to_optics_list()

        # Assert
        assert contains_instance(new_optics, AddConstant)

    @pytest.mark.xdist_group("2")
    @pytest.mark.software
    def test_pop_when_correct(self: object) -> None:
        # Arrange
        detector: object = TolimanDetector()

        # Act
        new_detector: object = detector.pop()
        new_optics: list = new_detector.to_optics_list()

        # Assert
        # TODO: Check that I actually have this correct and upgrade
        #       to make it more programmatic.
        assert not contains_instance(new_optics, ApplyPixelResponse)


class TestAlphaCentauri(object):
    @pytest.mark.xdist_group("3")
    @pytest.mark.software
    def test_constructor_when_not_given_spectrum(self: object) -> None:
        # Arrange
        alpha_centauri: object = AlphaCentauri()

        # Act/Assert
        assert isinstance(alpha_centauri.spectrum, CombinedSpectrum)

#    @pytest.mark.software
#    def test_constructor_when_csv_empty(self: object) -> None:
#
#    @pytest.mark.physical
#    def test_spectrum_in_filter_range(self: object) -> None:


# TODO: Make sure that the stars are correctly thinned randomly.
class TestBackground(object):
    @pytest.mark.xdist_group("4")
    @pytest.mark.software
    def test_constructor_when_all_used(self: object) -> None:
        # Arrange/Act
        background: object = Background()

        # TODO: Get the length of the csv and check against that.

        # Assert
        assert background.flux.size > 10

#    @pytest.mark.software
#    def test_constructor_when_csv_empty(self: object) -> None:
#        # TODO: Make an empty csv and read it in. 

    @pytest.mark.xdist_group("1")
    @pytest.mark.software
    def test_constructor_when_thinned(self: object) -> None:
        # Arrange/Act
        background: object = Background(number_of_bg_stars=10)

        # Assert
        assert background.flux.size == 10
