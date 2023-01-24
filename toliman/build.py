import os
import warnings
import tqdm
import requests

__author__ = "Jordan Dennis"

HOME: str = "toliman/assets/grid/phoenix"
PHOENIX_HOME: str = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/phoenix"
PHOENIXM00: str = "phoenixm00"
PHOENIXP03: str = "phoenixp03"
PHOENIX_PATHS: str = [
    "{}/{}_5200.fits".format(PHOENIXM00, PHOENIXM00),
    "{}/{}_5300.fits".format(PHOENIXM00, PHOENIXM00),
    "{}/{}_5700.fits".format(PHOENIXM00, PHOENIXM00),
    "{}/{}_5800.fits".format(PHOENIXM00, PHOENIXM00),
    "{}/{}_5200.fits".format(PHOENIXP03, PHOENIXP03),
    "{}/{}_5300.fits".format(PHOENIXP03, PHOENIXP03),
    "{}/{}_5700.fits".format(PHOENIXP03, PHOENIXP03),
    "{}/{}_5800.fits".format(PHOENIXP03, PHOENIXP03),
    "catalog.fits"
]

def _is_phoenix_installed() -> bool:
    """
    Check if "phoenix" is installed.

    Returns
    -------
    installed: bool
        True if all the phoenix files are present else false.
    """
    if not os.path.exists(HOME):
        return False

    for path in PHOENIX_PATHS:
        rel_path: str = "{}/{}".format(HOME, path)
        if not os.path.isfile(path):
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
        for path in _accumulate_path(HOME.split("/")):
            if not os.path.exists(path):
                os.mkdir(path)

    for path in [PHOENIXM00, PHOENIXP03]:
        rel_path: str = "{}/{}".format(HOME, path)
        if not os.path.exists(rel_path):
            os.mkdir(rel_path)

    for file in PHOENIX_PATHS:
        path: str = "{}/{}".format(HOME, file)
    
        if not os.path.isfile(path):
            url: str = "{}/{}".format(PHOENIX_HOME, file)

            with open(path, "wb") as file_dev:
                url: str = "{}/{}".format(PHOENIX_HOME, file)
                response: iter = requests.get(url, stream=True)
                total_size: int = int(response.headers.get('content-length', 0))

                print("Downloading: {}.".format(url))

                progress: object = tqdm.tqdm(
                    total=total_size,
                    unit='iB', 
                    unit_scale=True
                )

                for data in response.iter_content(1024):
                    progress.update(len(data))
                    file_dev.write(data)

def main():
    print("Building...")
    if not _is_phoenix_installed():
        print("Installing phoenix!")
        _install_phoenix()


main()
