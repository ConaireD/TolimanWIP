import pytest

class TestTolimanOptics(object):
    def test___init__(self: object) -> None:
        # Operate in static mode

        # Load a mask that is too large
        with pytest.expect(NotImplementedError):

        # Mask sampling error

        # Incorrect file address error.

        # Simulate aberrations (yes and no)

        # Simulate polish

        # Operate in Fresnel mode
        with pytest.expect(NotImplementedError):


    def test_insert(self: object) -> None:
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
