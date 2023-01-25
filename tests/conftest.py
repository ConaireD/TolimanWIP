import os 

ASSETS: str = "tests/assets"
PHOENIX: str = "{}/grid/phoenix".format(ASSETS)
PHOENIXS: str = ["phoenixm00", "phoenixp03"]
NUMBERS: list = [5200, 5300, 5700, 5800]
PHOENIXS: list = [
    "{}/{}/{}_{}.fits".format(PHOENIX, phoe, phoe, num) 
        for num in NUMBERS 
        for phoe in PHOENIXS
]

def setup_assets() -> None:
    # Creating files so that they are skipped on install.
    for path in PHOENIXS:
        if not os.path.exists(path):
            with open(path, "w") as file:
                continue

    for path in PHOENIXS:
        if not os.path.isfile(path):
            raise ValueError("Failed to create paths.")

def _teardown_assets() -> None:
    os.rmdir(ASSETS)    
