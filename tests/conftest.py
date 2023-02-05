import os 

ASSETS: str = "tests/assets"
PHOENIX: str = "{}/grid/phoenix".format(ASSETS)
PHOENIXS: str = ["phoenixm00", "phoenixp03"]
NUMBERS: list = [5200, 5300, 5700, 5800]
PHOENIX_FILES: list = [
    "{}/{}/{}_{}.fits".format(PHOENIX, phoe, phoe, num) 
        for num in NUMBERS 
        for phoe in PHOENIXS
]

def setup_assets() -> None:
    # Creating files so that they are skipped on install.
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

def _teardown_assets() -> None:
    os.rmdir(ASSETS)    
