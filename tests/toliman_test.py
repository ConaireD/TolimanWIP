import pytest
import dLux as dl

from toliman import (
    TolimanOptics,
    TolimanDetector,
    AlphaCentauri,
    Background
)

class TestTolimanOptics(object):
    def test_constructor_when_static(self: object) -> None:
        # Arrange/Act
        static_toliman: object = TolimanOptics(
            operate_in_static_mode = True
        )

        # Assert
        optics: list = static_toliman.to_optics_list() 
        assert _contains_optic(optics, dl.StaticAperture)


    def test_constructor_when_not_static(self: object) -> None:
        # Arrange/Act
        dynamic_toliman: object = TolimanOptics(
            operate_in_static_mode = False
        )

        # Assert
        optics: list = dynamic_toliman.to_optics_list() 
        assert not _contains_optic(optics, dl.StaticAperture)


    def test_constructor_when_mask_too_large(self: object) -> None:
        with pytest.expect(NotImplementedError):
            # Arrange/Act/Assert
            toliman: object = TolimanOptics(
                pixels_in_pupul = 2048
            ) 


    def test_constructor_when_mask_incorrectly_sampled(self: object) -> None:
        # Arrange/Act/Assert
        with pytest.expect(ValueError):
            toliman: object = TolimanOptics(
                pixels_in_pupil = 125
            )


    def test_constructor_when_mask_is_correct(self: object) -> None:
        # Arrange/Act
        toliman: object = TolimanOptics(pixels_in_pupil = 256)

        # Assert
        optics: list = toliman.to_optics_list() 
        assert _contains_optic(optics, dl.ApplyOPD)


    def test_constructor_when_mask_is_correct_at_max(self: object) -> None:
        # Arrange/Act
        toliman: object = TolimanOptics(pixels_in_pupil = 1024)

        # Assert
        optics: list = toliman.to_optics_list() 
        assert _contains_optic(optics, dl.ApplyOPD)


    def test_constructor_when_mask_file_is_incorrect(self: object) -> None:
        # Incorrect file address error.
        with pytest.expect(ValueError):
            toliman: object = TolimanOptics(
                path_to_mask = "i/don't/exist.npy"
            )

    def test_constructor_when_aberrated(self: object) -> None:
        # Arrange/Act
        toliman: object = TolimanOptics(simulate_aberrations = True)
        opitcs: list = toliman.to_optics_list()

        # Assert
        assert _contains_instance(optics, dl.StaticAberratedAperture)

    def test_constructor_when_not_aberrated(self: object) -> None:
        toliman: object = TolimanOptics(simulate_aberrations = False)
        opitcs: list = list(toliman.layers.values())
        assert not _contains_instance(optics, dl.StaticAberratedAperture)

    def test_constructor_when_polish_is_simulated(self: object) -> None:
        # Simulate polish
        with pytest.expect(NotImplementedError):
            toliman: object = TolimanOptics(
                simulate_polish = True
            )

    def test_constructor_when_polish_is_not_simulated(self: object) -> None:

    def test_constructor_when_using_fresnel(self: object) -> None:
        # Operate in Fresnel mode
        with pytest.expect(NotImplementedError):
            toliman: object = TolimanOptics(operate_in_fresnel_mode = True)

    def test_constructor_when_not_using_fresnel(self: object) -> None:

    def test_insert(self: object) -> None:
        # Test not an optical layer
        toliman: object = TolimanOptics()
        with pytest.expect(ValueError):
            toliman.insert(0, 1)

        optics: list = list(toliman.layers.values())
        length: int = len(optics)

        insertion: object = dl.HexagonalAperture(1.)
        toliman: object = toliman.insert(0, insertion)

        new_optics: list = list(toliman.layers.values())

            

    def test_remove(self: object) -> None:


def TestTolimanDetector(object):
    def test___init__(self: object) -> None:
        # Simulate Jitter
        # Extra dectector layers already contains a Jitter

        # Simulate saturation 
        # Extra detector layers already contains a Saturation 

        # Simulate pixel response
        # Extra detector layers already contains a PixelResponse
        
        # Empty detector

    def test_insert(self: object) -> None:
    def test_remove(self: object) -> None:


class TestAlphaCentauri(object):
    def test___init__(self: object) -> None:
        # Test without spectrum 
        # test wth spectrum


class TestBackground(object):
    def test___init__(self: object) -> None:
        # Test without spectrum 
        # test wth spectrum
