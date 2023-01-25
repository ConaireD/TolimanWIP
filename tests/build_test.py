import pytest

# TODO: I need better messages for this case
def test_is_phoenix_installed_when_installed():
def test_is_phoenix_installed_when_uninstalled():
# TODO: What behaviour should this exhibit.
def test_is_phoenix_installed_when_partially_installed():

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

