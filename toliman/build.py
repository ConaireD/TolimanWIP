import os
import warnings

__author__ = "Jordan Dennis"

HOME: str = "toliman/assets/grid/phoenix"
PHOENIX_HOME: str = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/phoenix/"
PHOENIX_PATHS: str = [
    "phoenixm00/phoenixm00_5200.fits"
    "phoenixm00/phoenixm00_5300.fits"
    "phoenixm00/phoenixm00_5700.fits"
    "phoenixm00/phoenixm00_5800.fits"
    "phoenixp03/phoenixp03_5200.fits"
    "phoenixp03/phoenixp03_5300.fits"
    "phoenixp03/phoenixp03_5700.fits"
    "phoenixp03/phoenixp03_5800.fits"
    "catalog.fits"
]

def _is_phoenix_installed() -> bool:
    if not os.path.exists(HOME):
        return False

    for path in PHOENIX_PATHS:
        if not os.is_file(path):
            return False

    return True


def _accumulate_path(strings: list, paths: str = []) -> list:
    if not strings:
        return paths
    else:
        if not paths:
            paths.append(strings.pop(0))
        else:
            paths.append(paths[-1] + "/" + strings.pop(0))
        return _accumulate_path(strings, paths)

def _install_phoenix() -> bool:
    if not os.path.exists(HOME):
        for 
        if not os.path.exists("toliman"):
            os.mkdir("toliman")
        
        if not os.path.exists("toliman/assets"):
            os.mkdir("toliman/assets")

        if not os.path.exists("toliman/assets/grid"):
            os.mkdir("toliman/assets/grid")

        if not os.path.exists("toliman/assets/grid/phoenix"):
            os.mkdir("toliman/assets/grid/phoenix")

    if not os.is_file()


