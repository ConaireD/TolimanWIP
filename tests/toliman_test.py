import pytest
import dLux as dl

from toliman import (
    TolimanOptics,
    TolimanDetector,
    AlphaCentauri,
    Background
)

class TestTolimanOptics(object):
    def test___init__(self: object) -> None:
        # Operate in static mode
        static_toliman: object = TolimanOptics(
            operate_in_static_mode = True
        )

        optics: list = list(static_toliman.layers.values())
        assert _contains_optic(optics, dl.StaticAperture)

        # Load a mask that is too large
        with pytest.expect(NotImplementedError):
            toliman: object = TolimanOptics(
                pixels_in_pupul = 2048
            ) 

        # Mask sampling error
        with pytest.expect(ValueError):
            toliman: object = TolimanOptics(
                pixels_in_pupil = 125
            )

        # Incorrect file address error.
        with pytest.expect(ValueError):
            toliman: object = TolimanOptics(
                path_to_mask = "i/don't/exist.npy"
            )

        # Simulate aberrations (yes and no)
        toliman: object = TolimanOptics(simulate_aberrations = True)
        opitcs: list = list(toliman.layers.values())
        assert _contains_instance(optics, dl.StaticAberratedAperture)

        toliman: object = TolimanOptics(simulate_aberrations = False)
        opitcs: list = list(toliman.layers.values())
        assert not _contains_instance(optics, dl.StaticAberratedAperture)

        # Simulate polish
        with pytest.expect(NotImplementedError):
            toliman: object = TolimanOptics(
                simulate_polish = True
            )

        # Operate in Fresnel mode
        with pytest.expect(NotImplementedError):
            toliman: object = TolimanOptics(operate_in_fresnel_mode = True)


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
