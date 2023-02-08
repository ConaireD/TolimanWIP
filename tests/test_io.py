import toliman.io as io

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
        create_fake_csv: fixture[str],
        remove_installation: fixture[None],
    ) -> None:
    """
    """
    array: float = io.read_csv_to_jax_array(create_fake_csv)
    assert array.shape == (number_of_rows, number_of_columns)

