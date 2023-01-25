import pytest

# TODO: I need better messages for this case
def test_is_phoenix_installed_when_installed():
    # Arrange
    if not os.path.exists(PHOENIX):
        for path in _accumulate_path(PHOENIX):
            os.mkdir(path)

    for phoenix in PHOENIXS:
        path: str = "{}/{}".format(PHOENIX, phoenix)
        if not os.path.exists(path):
            os.mkdir(path)

    for path in PHOENIX_FILES:
        if not os.path.exists(path):
            with open(path, "w") as file:
                continue

    for path in PHOENIX_FILES:
        if not os.path.isfile(path):
            raise ValueError("Failed to create paths.")

    # Act 
    # TODO: Add the root kwarg
    installed: bool = _is_phoenix_installed(ASSETS)

    # assert
    assert installed

    # Clean up
    if os.path.exists(assets)
        os.rmdir(assets)

def test_is_phoenix_installed_when_uninstalled():
    # Arange
    if os.path.exists(ASSETS)
        os.rmdir(ASSETS)

    # Act
    installed: bool = _is_phoenix_installed(ASSETS)

    # Assert
    assert not installed


# TODO: What behaviour should this exhibit.
def test_is_phoenix_installed_when_partially_installed():
    # Arrange
    if not os.path.exists(PHOENIX):
        for path in _accumulate_path(PHOENIX):
            os.mkdir(path)

    # Partial fake install
    path: str = "{}/{}".format(PHOENIX, PHOENIXS[0])
    if not os.path.exists(path):
        os.mkdir(path)

    for path in PHOENIX_FILES:
        if path.find(PHOENIXS[0]) > 0:
            if not os.path.exists(path):
                with open(path, "w") as file:
                    continue
                if not os.path.isfile(path):
                    raise ValueError("Failed to create paths.")

    # Act
    installed: bool = _is_phoenix_installed(ASSETS)

    # Assert
    assert not installed

    # Clean Up
    if os.path.exists(assets)
        os.rmdir(assets)

   
    

# TODO: This should be parametrisable
def test_install_phoenix_creates_file():
# TODO: This should be parametrisable
def test_install_phoenix_file_is_readable():

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

