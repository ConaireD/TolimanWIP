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
        """
        Does static operation use a static aperture?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        static_toliman: object = TolimanOptics(operate_in_static_mode=True)
        optics: list = static_toliman.to_optics_list()
        assert contains_instance(optics, StaticAperture)

    @pytest.mark.xdist_group("1")
    @pytest.mark.software
    def test_constructor_when_not_static(self: object) -> None:
        """
        Does dynamic operation not include a static aperture.

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        dynamic_toliman: object = TolimanOptics(operate_in_static_mode=False)
        optics: list = dynamic_toliman.to_optics_list()
        assert not contains_instance(optics, StaticAperture)

    @pytest.mark.xdist_group("2")
    @pytest.mark.software
    def test_constructor_when_mask_too_large(self: object) -> None:
        """
        Does a resolution higher than the masks raise an error?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        with pytest.raises(NotImplementedError):
            toliman: object = TolimanOptics(pixels_in_pupil=2048)

    @pytest.mark.xdist_group("3")
    @pytest.mark.software
    def test_constructor_when_mask_incorrectly_sampled(self: object) -> None:
        """
        If resolution is not a factor of 1024 does it raise an error?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        with pytest.raises(ValueError):
            toliman: object = TolimanOptics(pixels_in_pupil=125)

    @pytest.mark.xdist_group("1")
    @pytest.mark.software
    def test_constructor_when_mask_is_correct(self: object) -> None:
        """
        Does a possible resolution create an appropriate mask?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        toliman: object = TolimanOptics(pixels_in_pupil=256)
        optics: list = toliman.to_optics_list()
        assert contains_instance(optics, AddOPD)

    @pytest.mark.xdist_group("4")
    @pytest.mark.software
    @pytest.mark.skipif(not os.path.isfile(MASK_DIR), reason="The mask is not installed.")
    def test_constructor_when_mask_is_correct_at_max(self: object) -> None:
        """
        Can the program run at the mask resolution?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        mask: float = np.load(MASK_DIR)
        max_size: int = mask.shape[0]
        toliman: object = TolimanOptics(pixels_in_pupil=max_size)
        optics: list = toliman.to_optics_list()
        assert contains_instance(optics, AddOPD)

    @pytest.mark.xdist_group("1")
    @pytest.mark.software
    @pytest.mark.skipif(os.path.isfile("i/don't/exist.npy"), reason="File exists.")
    def test_constructor_when_mask_file_is_incorrect(self: object) -> None:
        """
        Does an incorrect path to the mask raise an error?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        with pytest.raises(ValueError):
            toliman: object = TolimanOptics(path_to_mask="i/don't/exist.npy")

    @pytest.mark.xdist_group("4")
    @pytest.mark.software
    def test_constructor_when_aberrated(self: object) -> None:
        """
        Are aberrations simulated when requested?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        toliman: object = TolimanOptics(simulate_aberrations=True)
        optics: list = toliman.to_optics_list()
        assert contains_instance(optics, StaticAberratedAperture)

    @pytest.mark.xdist_group("2")
    @pytest.mark.software
    def test_constructor_when_not_aberrated(self: object) -> None:
        """
        Are aberrations simulated when ignored?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        toliman: object = TolimanOptics(simulate_aberrations=False)
        optics: list = list(toliman.layers.values())
        assert not contains_instance(optics, StaticAberratedAperture)

    @pytest.mark.xdist_group("3")
    @pytest.mark.software
    def test_constructor_when_polish_is_simulated(self: object) -> None:
        """
        Does a request for mirror polish raise an error?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        with pytest.raises(NotImplementedError):
            toliman: object = TolimanOptics(simulate_polish=True)

    @pytest.mark.xdist_group("4")
    @pytest.mark.software
    def test_constructor_when_polish_is_not_simulated(self: object) -> None:
        """
        Is mirror polished ignored when not requested?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        toliman: object = TolimanOptics(simulate_polish=False)
        optics: list = toliman.to_optics_list()
        assert not contains_instance(optics, GeometricAberrations)

    @pytest.mark.xdist_group("3")
    @pytest.mark.software
    def test_constructor_when_using_fresnel(self: object) -> None:
        """
        Does requesting Fresnel raise and error?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        with pytest.raises(NotImplementedError):
            toliman: object = TolimanOptics(operate_in_fresnel_mode=True)

    @pytest.mark.xdist_group("1")
    @pytest.mark.software
    def test_constructor_when_not_using_fresnel(self: object) -> None:
        """
        Does not requesting Fresnel succeed?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        toliman: object = TolimanOptics(operate_in_fresnel_mode=False)
        optics: list = toliman.to_optics_list()
        assert not contains_instance(optics, FresnelPropagator)

    @pytest.mark.xdist_group("4")
    @pytest.mark.software
    def test_insert_when_type_is_incorrect(self: object) -> None:
        """
        Does insertion of an incorrect type fail?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        toliman: object = TolimanOptics()
        with pytest.raises(ValueError):
            toliman.insert(0, 1)

    @pytest.mark.xdist_group("1")
    @pytest.mark.software
    def test_insert_when_index_is_too_long(self: object) -> None:
        """
        Does insertion after the end of the list fail?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        toliman: object = TolimanOptics()
        optics: list = toliman.to_optics_list()
        length: int = len(optics)
        wrong_index: int = length + 1
        element_to_insert: object = CircularAperture(1.0)
        with pytest.raises(ValueError):
            toliman.insert(wrong_index, element_to_insert)

    @pytest.mark.xdist_group("2")
    @pytest.mark.software
    def test_insert_when_index_is_negative(self: object) -> None:
        """
        Does insertion using negative indices fail?
        
        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        toliman: object = TolimanOptics()
        element_to_insert: object = CircularAperture(1.0)
        with pytest.raises(ValueError):
            toliman.insert(-1, element_to_insert)

    @pytest.mark.xdist_group("3")
    @pytest.mark.software
    def test_insert_when_correct(self: object) -> None:
        """
        Does insert work when the pre-conditions are met?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        toliman: object = TolimanOptics()
        optics: list = toliman.to_optics_list()
        length: int = len(optics)
        insertion: object = HexagonalAperture(1.0)
        toliman: object = toliman.insert(insertion, 0)
        new_optics: list = toliman.to_optics_list()
        assert contains_instance(new_optics, HexagonalAperture)

    @pytest.mark.xdist_group("4")
    @pytest.mark.software
    def test_remove_when_index_is_too_long(self: object) -> None:
        """
        Is it impossible to remove elements past the end of the list?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        toliman: object = TolimanOptics()
        optics: list = toliman.to_optics_list()
        length: int = len(optics)
        wrong_index: int = length + 1
        with pytest.raises(ValueError):
            toliman.remove(wrong_index)

    @pytest.mark.xdist_group("1")
    @pytest.mark.software
    def test_remove_when_index_is_negative(self: object) -> None:
        """
        Is it possible to remove elements using negative indices?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        toliman: object = TolimanOptics()
        with pytest.raises(ValueError):
            toliman.remove(-1)

    @pytest.mark.xdist_group("2")
    @pytest.mark.software
    def test_remove_when_correct(self: object) -> None:
        """
        Can elements be removed when the pre-conditions are met?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        toliman: object = TolimanOptics()
        optics: list = toliman.to_optics_list()
        new_toliman: object = toliman.remove(0)
        new_optics: list = new_toliman.to_optics_list()
        assert not contains_instance(new_optics, CreateWavefront)

    @pytest.mark.xdist_group("3")
    @pytest.mark.software
    @pytest.mark.parametrize("atype", [str, float, int])
    def test_append_when_type_is_incorrect(self: object, atype: type) -> None:
        """
        Is it impossible to append the incorrect type?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        toliman: object = TolimanOptics()
        with pytest.raises(ValueError):
            toliman.append(atype(1))

    @pytest.mark.xdist_group("4")
    @pytest.mark.software
    def test_append_when_correct(self: object) -> None:
        """
        Is it possible to append an element of the correct type?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        toliman: object = TolimanOptics()
        element_to_append: object = CircularAperture(1.0)
        toliman: object = toliman.append(element_to_append)
        optics: list = toliman.to_optics_list()
        assert contains_instance(optics, CircularAperture)

    @pytest.mark.xdist_group("1")
    @pytest.mark.software
    def test_pop_removes_element(self: object) -> None:
        """
        Does pop remove the last element?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        toliman: object = TolimanOptics()
        optics: list = toliman.to_optics_list()
        new_toliman: object = toliman.pop()
        new_optics: list = new_toliman.to_optics_list()
        assert optics[-1] != new_optics[-1]


class TestTolimanDetector(object):
    @pytest.mark.xdist_group("2")
    @pytest.mark.software
    def test_constructor_when_jittered(self: object) -> None:
        """
        Is jitter included on request?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        detector: object = TolimanDetector(simulate_jitter=True)
        optics: list = detector.to_optics_list()
        assert contains_instance(optics, ApplyJitter)

    @pytest.mark.xdist_group("3")
    @pytest.mark.software
    def test_constructor_when_not_jittered(self: object) -> None:
        """
        Is jitter excluded when not requested?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        detector: object = TolimanDetector(simulate_jitter=False)
        optics: list = detector.to_optics_list()
        assert not contains_instance(optics, ApplyJitter)

    @pytest.mark.xdist_group("4")
    @pytest.mark.software
    def test_constructor_when_jitter_is_repeated(self: object) -> None:
        """
        Does adding jitter twice raise an error?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        jitter: object = ApplyJitter(2.0)
        with pytest.raises(ValueError):
            detector: object = TolimanDetector(
                simulate_jitter=True, extra_detector_layers=[jitter]
            )

    @pytest.mark.xdist_group("1")
    @pytest.mark.software
    def test_constructor_when_saturated(self: object) -> None:
        """
        Is saturation included when requested?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        detector: object = TolimanDetector(simulate_saturation=True)
        optics: list = detector.to_optics_list()
        assert contains_instance(optics, ApplySaturation)

    @pytest.mark.xdist_group("2")
    @pytest.mark.software
    def test_constructor_when_not_saturated(self: object) -> None:
        """
        Is saturation excluded when not requested?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        detector: object = TolimanDetector(simulate_saturation=False)
        optics: list = detector.to_optics_list()
        assert not contains_instance(optics, ApplySaturation)

    @pytest.mark.xdist_group("3")
    @pytest.mark.software
    def test_constructor_when_saturation_is_repeated(self: object) -> None:
        """
        Does repetition of saturation cause an error?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        saturation: object = ApplySaturation(2)
        with pytest.raises(ValueError):
            detector: object = TolimanDetector(
                simulate_saturation=True, extra_detector_layers=[saturation]
            )

    @pytest.mark.xdist_group("4")
    @pytest.mark.software
    def test_constructor_when_pixels_respond(self: object) -> None:
        """
        Is pixel response included on request?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        detector: object = TolimanDetector(simulate_pixel_response=True)
        optics: list = detector.to_optics_list()
        assert contains_instance(optics, ApplyPixelResponse)

    @pytest.mark.xdist_group("1")
    @pytest.mark.software
    def test_constructor_when_pixels_dont_respond(self: object) -> None:
        """
        Is pixel response excluded on request.

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        detector: object = TolimanDetector(simulate_pixel_response=False)
        optics: list = detector.to_optics_list()
        assert not contains_instance(optics, ApplyPixelResponse)

    @pytest.mark.xdist_group("2")
    @pytest.mark.software
    def test_constructor_when_pixel_response_is_repeated(self: object) -> None:
        """
        Does repetition of pixel response raise and error?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        pixel_response: object = ApplySaturation(2)
        with pytest.raises(ValueError):
            detector: object = TolimanDetector(
                simulate_pixel_response=True, extra_detector_layers=[pixel_response]
            )

    @pytest.mark.xdist_group("3")
    @pytest.mark.software
    def test_constructor_when_correct(self: object) -> None:
        """
        Does the default constructor work?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        detector: object = TolimanDetector()
        optics: list = detector.to_optics_list()
        assert contains_instance(optics, ApplyJitter)
        assert contains_instance(optics, ApplySaturation)
        assert contains_instance(optics, ApplyPixelResponse)

    @pytest.mark.xdist_group("4")
    @pytest.mark.software
    def test_constructor_when_empty(self: object) -> None:
        """
        Does an empty detector raise an error?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        with pytest.raises(ValueError):
            detector: object = TolimanDetector(
                simulate_jitter=False,
                simulate_saturation=False,
                simulate_pixel_response=False,
            )

    @pytest.mark.xdist_group("1")
    @pytest.mark.software
    def test_insert_when_type_is_incorrect(self: object) -> None:
        """
        Does insert only work for DetectorLayers?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        detector: object = TolimanDetector()
        with pytest.raises(ValueError):
            detector.insert(0, 1)

    @pytest.mark.xdist_group("2")
    @pytest.mark.software
    def test_insert_when_index_is_too_long(self: object) -> None:
        """
        Does insert fail beyond the list?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        detector: object = TolimanDetector()
        optics: object = detector.to_optics_list()
        length: int = len(optics)
        too_long: int = length + 1
        element_to_insert: object = AddConstant(1.0)
        with pytest.raises(ValueError):
            detector.insert(too_long, element_to_insert)

    @pytest.mark.xdist_group("3")
    @pytest.mark.software
    def test_insert_when_index_is_negative(self: object) -> None:
        """
        Does insert fail on negative indices?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        detector: object = TolimanDetector()
        element_to_insert: object = AddConstant(1.0)
        with pytest.raises(ValueError):
            detector.insert(-1, element_to_insert)

    @pytest.mark.xdist_group("4")
    @pytest.mark.software
    def test_insert_when_correct(self: object) -> None:
        """
        Does insert work when the preconditions are met?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        detector: object = TolimanDetector()
        element_to_insert: object = AddConstant(1.0)
        detector: object = detector.insert(element_to_insert, 0)
        optics: list = detector.to_optics_list()
        assert contains_instance(optics, AddConstant)

    @pytest.mark.xdist_group("1")
    @pytest.mark.software
    def test_remove_when_index_is_too_long(self: object) -> None:
        """
        Does remove fail beyond the list?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        detector: object = TolimanDetector()
        optics: list = detector.to_optics_list()
        length: int = len(optics)
        too_long: int = length + 1
        with pytest.raises(ValueError):
            detector.remove(too_long)

    @pytest.mark.xdist_group("2")
    @pytest.mark.software
    def test_remove_when_index_is_negative(self: object) -> None:
        """
        Does remove fail when the index is negative?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        detector: object = TolimanDetector()
        with pytest.raises(ValueError):
            detector.remove(-1)

    @pytest.mark.xdist_group("3")
    @pytest.mark.software
    def test_remove_when_correct(self: object) -> None:
        """
        Does remove succeed when the pre-conditions are met?
    
        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        detector: object = TolimanDetector()
        detector: object = detector.remove(0)
        optics: list = detector.to_optics_list()
        assert not contains_instance(optics, ApplyJitter)

    @pytest.mark.xdist_group("4")
    @pytest.mark.software
    def test_append_when_type_is_incorrect(self: object) -> None:
        """
        Does append only work for DetectorLayers?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        detector: object = TolimanDetector()
        with pytest.raises(ValueError):
            detector.append(1)

    @pytest.mark.xdist_group("1")
    @pytest.mark.software
    def test_append_when_correct(self: object) -> None:
        """
        Does append work when the preconditions are met?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        detector: object = TolimanDetector()
        element_to_insert: object = AddConstant(2.0)
        new_detector: object = detector.append(element_to_insert)
        new_optics: list = new_detector.to_optics_list()
        assert contains_instance(new_optics, AddConstant)

    @pytest.mark.xdist_group("2")
    @pytest.mark.software
    def test_pop_when_correct(self: object) -> None:
        """
        Does pop work when the pre-conditions are met?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        detector: object = TolimanDetector()
        new_detector: object = detector.pop()
        new_optics: list = new_detector.to_optics_list()
        assert not contains_instance(new_optics, ApplyPixelResponse)


class TestAlphaCentauri(object):
    @pytest.mark.xdist_group("3")
    @pytest.mark.software
    def test_constructor_when_not_given_spectrum(self: object) -> None:
        """
        Does AlphaCentauri create a default spectrum?

        Marks
        -----
        xdist_group: str
            The named process to run the test on.
        software: None
            Tests an implementation detail not physics.
        """
        alpha_centauri: object = AlphaCentauri()
        assert isinstance(alpha_centauri.spectrum, CombinedSpectrum)


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
