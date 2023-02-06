import toliman.math as math

def test_normalise_in_range():
    # Arrange
    array: float = np.arange(10)

    # Act
    narray: float = math.normalise(array)

    # Assert
    assert (0.0 <= narray <= 1.0).all()
