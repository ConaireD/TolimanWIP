import toliman.collections as collections
import pytest

@pytest.mark.parametrize("instance_of,collection", 
    [
        (float, [0.0]), 
        (float, [0, 0.0]),
        (float, [0.0, 0, "str"]),
        (float, ["str", 0.0, "str", 0.0, 0]),
        (str, ["str"]),
        (str, ["str", 0]),
        (str, [0, "str", 0.0]),
        (str, ["str", 0.0, 0, "str"]),
        (int, [0]),
        (int, [0, 0]),
        (int, [0, "str", 0]),
        (int, ["str", 0.0, 0.0, 0]),
    ]
)
def test_contains_instance_when_type_present(
        instance_of: type, 
        collection: list
    ) -> None:
    """
    """
    assert collections.contains_instance(collection, instance_of)

@pytest.mark.parametrize("instance_of,collection",
    [
        (float, [0]),
        (float, [0, "str"]),
        (float, ["str", 0, 0, "str"]),
        (str, [0]),
        (str, [0, 0.0]),
        (str, [0.0, 0.0, 0, 0]),
        (int, ["str"]),
        (int, ["str", 0.0]),
        (int, [0.0, 0.0, "str", "str"]),
    ]
)
def test_contains_instance_when_type_is_not_present(
        instance_of: type,
        collection: list,
    ) -> None:
    """
    """
    assert not collections.contains_instance(collection, instance_of)
