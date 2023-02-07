import pytest

@pytest.fixture(scope = "module")
def im_a_fixture() -> None:
    return "a"

@pytest.mark.parametrize("a", [im_a_fixture()])
def test_a(a: str, im_a_fixture) -> None:
    assert im_a_fixture == "a"
