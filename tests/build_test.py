import pytest

def make_phoenix_root_directory() -> None:
    if not os.path.exists(PHOENIX):
        for path in _accumulate_path(PHOENIX):
            os.mkdir(path)

def make_phoenix_type_directory(phoenix: str) -> None:
    path: str = "{}/{}".format(PHOENIX, phoenix)
    if not os.path.exists(path):
        os.mkdir(path)

def make_phoenix_type_files(phoenix: str) -> None:
    for number in NUMBERS:
        path: str = "{}/{}/{}_{}.fits".format(PHOENIX, phoenix, phoenix, number)
        if not os.path.exists(path):
            with open(path, "w") as file:
                continue
            if not os.path.isfile(path):
                raise ValueError

def remove_phoenix() -> None:
    if os.path.exists(assets)
        os.rmdir(assets)

def test_is_phoenix_installed_when_installed():
    # Arrange
    make_phoenix_root_directory()
    for phoenix in PHOENIXS:
        make_phoenix_type_directory(phoenix)
        make_phoenix_type_files(phoenix)

    # Act 
    # TODO: Add the root kwarg
    installed: bool = _is_phoenix_installed(ASSETS)
    remove_phoenix()

    # assert
    assert installed

def test_is_phoenix_installed_when_uninstalled():
    # Arange
    remove_phoenix()

    # Act
    installed: bool = _is_phoenix_installed(ASSETS)

    # Assert
    assert not installed

def test_is_phoenix_installed_when_partially_installed():
    # Arrange
    make_phoenix_root_directory()
    make_phoenix_type_directory(PHOENIXS[0])
    make_phoenix_type_files(PHOENIXS[0])

    # Act
    installed: bool = _is_phoenix_installed(ASSETS)

    # Assert
    assert not installed

    # Clean Up
    remove_phoenix()

@pytest.mark.parametrise("file", PHOENIX_FILES)
def test_install_phoenix_creates_file(file: str) -> None:
    # Arrange
    if not _is_phoenix_installed(ASSETS):
        _install_phoenix(ASSETS)

    # Act/Assert
    assert os.path.isfile(file)

    # TODO: Implement some cleanup
    if file == PHOENIX_FILES[-1]:
        remove_phoenix()

@pytest.mark.parametrise("file", PHOENIX_FILES)
def test_install_phoenix_file_is_readable(file: str):
    # Arrange
    if not _is_phoenix_installed(ASSETS):
        _install_phoenix(ASSETS)

    import astropy

    # Act/assert
    try:
        with astropy.io.fits.open(file) as fits:
            assert True
    except IOError as ioe:
        assert False

    # Clean up
    if file == PHOENIX_FILES[-1]:
        remove_phoenix()
    
def test_accumulate_path_when_empty():
# TODO: This should be parametrizable
def test_accumulate_path_when_correct():

# TODO: These should all only run if the first test passes.
def test_install_mask_creates_file():
def test_install_mask_is_readable():
def test_install_mask_is_correct_shape():

def test_is_mask_installed_when_false():
def test_is_mask_installed_when_true():

# TODO: Parametrize across multiple ndim.
def test_normalise_when_correct():
# TODO: Parametrize across a few different types.
def test_normalise_with_wrong_type():

def test_simulate_background_stars_has_correct_default_nrows():
def test_simulate_background_stars_has_correct_nrows():
def test_simulate_background_stars_has_correct_ncols():
def test_simulate_background_stars_is_numeric():

def test_simulate_alpha_cen_spectrum_has_correct_default_nrows():
def test_simulate_alpha_cen_spectrum_has_correct_nrows():
def test_simulate_alpha_cen_spectrum_has_correct_ncols():
def test_simulate_alpha_cen_spectrum_is_numeric():

def test_build_makes_background_csv():
def test_build_makes_spectra_csv():
def test_build_makes_phoenix_when_not_installed():
def test_build_doesnt_make_phoenix_when_installed():
def test_build_makes_mask_npy_when_not_installed():
def test_build_doesnt_make_mask_npy_when_installed()

