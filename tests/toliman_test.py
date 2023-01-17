import pytest
import dLux as dl

from dLux import (
    CircularAperture,
    HexagonalAperture,
    ApplyOPD,
    ApplyJitter,
    StaticAperture,
    StaticAberratedAperture
)

from toliman import TolimanOptics, TolimanDetector, AlphaCentauri, Background


class TestTolimanOptics(object):
    def test_constructor_when_static(self: object) -> None:
        # Arrange/Act
        static_toliman: object = TolimanOptics(operate_in_static_mode=True)

        # Assert
        optics: list = static_toliman.to_optics_list()
        assert _contains_optic(optics, dl.StaticAperture)

    def test_constructor_when_not_static(self: object) -> None:
        # Arrange/Act
        dynamic_toliman: object = TolimanOptics(operate_in_static_mode=False)

        # Assert
        optics: list = dynamic_toliman.to_optics_list()
        assert not _contains_optic(optics, dl.StaticAperture)

    def test_constructor_when_mask_too_large(self: object) -> None:
        with pytest.expect(NotImplementedError):
            # Arrange/Act/Assert
            toliman: object = TolimanOptics(pixels_in_pupul=2048)

    def test_constructor_when_mask_incorrectly_sampled(self: object) -> None:
        # Arrange/Act/Assert
        with pytest.expect(ValueError):
            toliman: object = TolimanOptics(pixels_in_pupil=125)

    def test_constructor_when_mask_is_correct(self: object) -> None:
        # Arrange/Act
        toliman: object = TolimanOptics(pixels_in_pupil=256)

        # Assert
        optics: list = toliman.to_optics_list()
        assert _contains_optic(optics, dl.ApplyOPD)

    def test_constructor_when_mask_is_correct_at_max(self: object) -> None:
        # Arrange/Act
        toliman: object = TolimanOptics(pixels_in_pupil=1024)

        # Assert
        optics: list = toliman.to_optics_list()
        assert _contains_optic(optics, dl.ApplyOPD)

    def test_constructor_when_mask_file_is_incorrect(self: object) -> None:
        # Incorrect file address error.
        # TODO: Make sure that this file does not exist
        #       This is a best practice thing as it can theoretically
        #       exist making this test environment dependent. This
        #       violates the R in the F.I.R.S.T principles.
        with pytest.expect(ValueError):
            toliman: object = TolimanOptics(path_to_mask="i/don't/exist.npy")

    def test_constructor_when_aberrated(self: object) -> None:
        # Arrange/Act
        toliman: object = TolimanOptics(simulate_aberrations=True)
        opitcs: list = toliman.to_optics_list()

        # Assert
        assert _contains_instance(optics, dl.StaticAberratedAperture)

    def test_constructor_when_not_aberrated(self: object) -> None:
        # Arrange/Act
        toliman: object = TolimanOptics(simulate_aberrations=False)
        opitcs: list = list(toliman.layers.values())

        # Assert
        assert not _contains_instance(optics, dl.StaticAberratedAperture)

    def test_constructor_when_polish_is_simulated(self: object) -> None:
        # Arrange/Act/Assert
        with pytest.expect(NotImplementedError):
            toliman: object = TolimanOptics(simulate_polish=True)

    def test_constructor_when_polish_is_not_simulated(self: object) -> None:
        # Arrange/Act
        toliman: object = TolimanOptics(simulate_polish=False)
        optics: list = toliman.to_optics_list()

        # Assert
        assert not _contains_instance(optics, GeometricAberrations)

    def test_constructor_when_using_fresnel(self: object) -> None:
        # Operate in Fresnel mode
        with pytest.expect(NotImplementedError):
            toliman: object = TolimanOptics(operate_in_fresnel_mode=True)

    def test_constructor_when_not_using_fresnel(self: object) -> None:
        # Arrange/Act
        toliman: object = TolimanOptics(operate_in_fresnel_mode=False)
        optics: list = toliman.to_optics_list()

        # TODO: get the correct name for the FresnelPropagator.
        # Assert
        assert not _contains_instance(optics, dl.FresnelPropagator)

    def test_insert_when_type_is_incorrect(self: object) -> None:
        # Arrange
        toliman: object = TolimanOptics()
        with pytest.expect(ValueError):
            # Act/Assert
            toliman.insert(0, 1)

    def test_insert_when_index_is_too_long(self: object) -> None:
        # Arrange
        toliman: object = TolimanOptics()
        optics: list = toliman.to_optics_list()
        length: int = len(optics)
        wrong_index: int = length + 1
        element_to_insert: object = dl.CircularAperture(1.0)

        # Act/Assert
        with pytest.expect(ValueError):
            toliman.insert(wrong_index, element_to_insert)

    def test_insert_when_index_is_negative(self: object) -> None:
        # Arrange
        toliman: object = TolimanOptics()
        element_to_insert: object = dl.CircularAperture(1.0)

        # Act/Assert
        with pytest.expect(ValueError):
            toliman.insert(-1, element_to_insert)

    def test_insert_when_correct(self: object) -> None:
        # Arrange
        toliman: object = TolimanOptics()
        optics: list = toliman.to_optics_list()
        length: int = len(optics)

        # Act
        insertion: object = dl.HexagonalAperture(1.0)
        toliman: object = toliman.insert(0, insertion)

        # Assert
        new_optics: list = toliman.to_optics_list()
        assert _contains_instance(new_optics, dl.HexagonalAperture)

    def test_remove_when_index_is_too_long(self: object) -> None:
        # Arrange
        toliman: object = TolimanOptics()
        optics: list = toliman.to_optics_list()
        length: int = len(optics)
        wrong_index: int = length + 1

        # Act/Assert
        with pytest.expect(ValueError):
            toliman.remove(wrong_index)

    def test_remove_when_index_is_negative(self: object) -> None:
        # Arrange
        toliman: object = TolimanOptics()

        # Act/Assert
        with pytest.expect(ValueError):
            toliman.remove(-1)

    def test_remove_when_correct(self: object) -> None:
        # Arrange
        toliman: object = TolimanOptics()
        optics: list = toliman.to_optics_list()

        # Act
        new_toliman: object = toliman.remove(0)
        new_optics: list = new_toliman.to_optics_list()

        # Assert
        assert not _contains_instance(new_optics, dl.StaticAperture)

    def test_append_when_type_is_incorrect(self: object) -> None:
        # Arrange
        toliman: object = TolimanOptics()

        # Act/Assert
        with pytest.expect(ValueError):
            toliman.append(1)

    def test_append_when_correct(self: object) -> None:
        # Arrange
        toliman: object = TolimanOptics()
        element_to_append: object = dl.CircularAperture(1.0)

        # Act
        toliman: object = toliman.append(element_to_append)
        optics: list = toliman.to_optics_list()

        # Assert
        assert _contains_instance(optics, dl.CircularAperture)

    def test_pop_removes_element(self: object) -> None:
        # Arrange
        toliman: object = TolimanOptics()
        optics: list = toliman.to_optics_list()

        # Act
        new_toliman: object = toliman.pop()
        new_optics: list = new_toliman.to_optics_list()

        # Assert
        assert optics[-1] != new_optics[-1]


def TestTolimanDetector(object):
    def test_constructor_when_jittered(self: object) -> None:
        # Arrange/Act 
        detector: object = TolimanDetector(simulate_jitter = True)
        optics: list = detector.to_optics_list()

        # Assert
        assert _contains_instance(optics, dl.ApplyJitter)


    def test_constructor_when_not_jittered(self: object) -> None:
        # Arrange/Act 
        detector: object = TolimanDetector(simulate_jitter = False)
        optics: list = detector.to_optics_list()

        # Assert
        assert not _contains_instance(optics, dl.ApplyJitter)


    def test_constructor_when_jitter_is_repeated(self: object) -> None:
        # Arrange
        jitter: object = ApplyJitter(2.)

        # Act/Assert
        with pytest.expect(ValueError):
            detector: object = TolimanDetector(simulate_jitter = True, extra_detector_layers = [jitter])


    def test_constructor_when_saturated(self: object) -> None:
        # Arrange/Act 
        detector: object = TolimanDetector(simulate_saturation = True)
        optics: list = detector.to_optics_list()

        # Assert
        assert _contains_instance(optics, dl.ApplySaturation)


    def test_constructor_when_not_saturated(self: object) -> None:
        # Arrange/Act 
        detector: object = TolimanDetector(simulate_saturation = False)
        optics: list = detector.to_optics_list()

        # Assert
        assert not _contains_instance(optics, dl.ApplySaturation)
        

    def test_constructor_when_saturation_is_repeated(self: object) -> None:
        # Arrange
        saturation: object = ApplySaturation(2)

        # Act/Assert
        with pytest.expect(ValueError):
            detector: object = TolimanDetector(simulate_saturation = True, extra_detector_layers = [saturation])


    def test_constructor_when_pixels_respond(self: object) -> None:
        # Arrange/Act 
        detector: object = TolimanDetector(simulate_pixel_response = True)
        optics: list = detector.to_optics_list()

        # Assert
        assert _contains_instance(optics, dl.ApplyPixelResponse)

        
    def test_constructor_when_pixels_dont_respond(self: object) -> None:
    def test_constructor_when_pixel_response_is_repeated(self: object) -> None:

    def test_constructor_when_correct(self: object) -> None:

    def test_insert_when_type_is_incorrect(self: object) -> None:
    def test_insert_when_index_is_too_long(self: object) -> None:
    def test_insert_when_index_is_negative(self: object) -> None:
    def test_insert_when_correct(self: object) -> None:

    def test_remove_when_index_is_too_long(self: object) -> None:
    def test_remove_when_index_is_negative(self: object) -> None:
    def test_remove_when_correct(self: object) -> None:


class TestAlphaCentauri(object):
    def test___init__(self: object) -> None:
        # Test without spectrum
        # test wth spectrum
        pass


class TestBackground(object):
    def test___init__(self: object) -> None:
        # Test without spectrum
        # test wth spectrum
        pass
