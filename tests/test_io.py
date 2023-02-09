import toliman.io as io
import pytest
import typing

class fixture(typing.Generic[typing.TypeVar("T")]): pass

@pytest.mark.parametrize("root", ["tmp", "dat"])
@pytest.mark.parametrize(
    "number_of_rows,number_of_columns",
    [
        (0, 3),
        (1, 5),
        (100, 3), 
    ]
)
def test_read_csv_to_jax_array_when_file_exists(
        number_of_rows: int,
        number_of_columns: int,
        make_fake_csv: fixture[str],
    ) -> None:
    """
    Fixtures
    --------
    make_fake_csv: fixture[str]
        Cleans the environment, writes a csv and cleans the environment. 

    Parameters
    ----------
    number_of_rows: int
        The number of rows in the csv/array. Indirectly parametrizes 
        make_fake_csv.
    number_of_columns: int
        The number of columns in the csv/array. Indirectly parametrizes 
        make_fake_csv.
    """
    array: float = io.read_csv_to_jax_array(make_fake_csv)
    shape: tuple = (number_of_rows,)
    if number_of_rows > 0: shape: tuple = shape + (number_of_columns,)
    assert array.shape == shape

