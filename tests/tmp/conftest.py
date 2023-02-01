import pytest 

@pytest.fixture
def inner(outer: callable) -> None:
    print("<inner></inner>")

@pytest.fixture
def outer() -> None:
    print("<outer>")
    yield
    print("</outer>")

