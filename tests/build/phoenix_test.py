import pytest
import toliman.build.phoenix as phoenix

# TODO: Make into fixtures
ASSETS: str = "tests/.assets"
PHOENIX: str = "{}/grid/phoenix".format(ASSETS)
PHOENIXS: str = ["phoenixm00", "phoenixp03"]
NUMBERS: list = [5200, 5300, 5700, 5800]
PHOENIX_FILES: list = [
    "{}/{}/{}_{}.fits".format(PHOENIX, phoe, phoe, num) 
        for num in NUMBERS 
        for phoe in PHOENIXS
]

def make_phoenix_root_directory() -> None:
    if not os.path.exists(PHOENIX):
        for path in _accumulate_path(PHOENIX.split("/")):
            if not os.path.exists(path):
                os.mkdir(path)

def make_phoenix_type_directory(phoenix: str) -> None:
    if not os.path.exists(PHOENIX):
        make_phoenix_root_directory()
    path: str = paths.concat([PHOENIX, phoenix])
    if not os.path.exists(path):
        os.mkdir(path)

def make_phoenix_type_files(phoenix: str) -> None:
    target: str = paths.concat([PHOENIX, phoenix])
    if not os.path.exists(target):
        make_phoenix_type_directory(phoenix)
    for number in NUMBERS:
        path: str = "{}/{}_{}.fits".format(target, phoenix, number)
        if not os.path.exists(path):
            with open(path, "w") as file:
                continue

def make_phoenix_catalog() -> None:
    if not os.path.exsists(PHOENIX)
        make_phoenix_root_directory()
    path: str = "{}/catalog.fits".format(PHOENIX)
    if not os.path.exists(path):
        with open(path, "w") as file:
            pass

def make_phoenix_installed() -> None:
    os.mkdir(ASSETS)
    make_phoenix_catalog()
    for phoenix in PHOENIXS:
        make_phoenix_type_files(phoenix)

def remove_phoenix() -> None:
    if os.path.exists(ASSETS):
        shutil.rmtree(ASSETS)

def test_is_phoenix_installed_when_fully_installed():
    # Arrange 
    make_phoenix_installed()

    # Assert
    assert phoenix.is_phoenix_installed()
    remove_phoenix()

def test_is_phoenix_installed_when_fully_installed():
    # Arrange 
    os.mkdir(ASSETS)
    make_phoenix_catalog()

    # Assert
    assert not phoenix.is_phoenix_installed()
    
def test_is_phoenix_installed_when_not_installed():
    # Arrange
    remove_phoenix()

    # Assert
    assert not phoenix.is_phoenix_installed()
