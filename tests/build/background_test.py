import toliman.build.background as bg
import toliman.constants as const

def test_load_background_stars_has_correct_shape():
    # Arrange
    bg_stars: float = bg.load_background_stars(0.0, 0.0, 2.0)

    # Act/Assert
    assert bg_stars.shape[0] == 3

def test_load_background_stars_within_cone():
    # Arrange
    rad: float = 2.0
    bg_stars: float = bg.load_background_stars(0.0, 0.0, rad)

    # Act/Assert
    hypot: callable = lambda x: np.hypot(x[0], x[1])
    hypots: float = np.apply_along_axis(hypot, 0, bg_stars[(0, 1)])
    assert (hypots <= rad).all()
