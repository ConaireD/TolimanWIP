import os

MASK_HOME: str = "https://github.com/ConaireD/TolimanWIP/raw/mask/mask.npy"

def _is_mask_installed() -> bool:
    """
    Check if the mask is installed.

    Returns
    -------
    installed: bool
        True is the mask is installed else false.
    """
    return os.path.isfile("{}/mask.npy".format(MASK))

def _install_mask() -> bool:
    """
    Install phoenix from the web.
    """
    MASK: str = "toliman/assets"
    if not os.path.exists(MASK):
        for path in _accumulate_path(MASK.split("/")):
            if not os.path.exists(path):
                os.mkdir(path)

    with open("{}/mask.npy".format(MASK), "wb") as file_dev:
        response: iter = requests.get(MASK_HOME, stream=True)
        total_size: int = int(response.headers.get('content-length', 0))

        print("Downloading: {}.".format(MASK_HOME))

        progress: object = tqdm.tqdm(
            total=total_size,
            unit='iB', 
            unit_scale=True
        )

        for data in response.iter_content(1024):
            progress.update(len(data))
            file_dev.write(data)

