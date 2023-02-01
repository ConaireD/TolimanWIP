import pytest 

@pytest.fixture
def inner(outer: callable) -> None:
    print("<inner>")
    yield
    print("</inner>")

@pytest.fixture
def outer() -> None:
    print("<outer>")
    yield
    print("</outer>")

