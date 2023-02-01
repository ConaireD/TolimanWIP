import os
import shutil
import toliman.constants as const

FILTER_MIN_WAVELENGTH: float = const.get_const_as_type("FILTER_MIN_WAVELENGTH", float)
FILTER_MAX_WAVELENGTH: float = const.get_const_as_type("FILTER_MAX_WAVELENGTH", float)

def create_fake_phoenix_installation(root: str, full = False) -> None:
    if os.path.isdir(root):
        shutil.rmtree(root)
    
    numbers: list = [5200, 5300, 5700, 5900]
    phoenix: str = "/".join([root, "grid", "phoenix"])
    phoenixm00: str = "/".join([phoenix, "phoenixm00"])
    phoenixp03: str = "/".join([phoenix, "phoenixp03"])
        
    os.makedirs(phoenixm00)
    os.makedirs(phoenixp03)

    open("{}/catalog.fits".format(phoenix)).close()
    for num in numbers:
        open("{}/phoenixm00_{}.fits".format(phoenixm00, num)).close()
        if full:
            open("{}/phoenixp03_{}.fits".format(phoenixm00, num)).close()
            
def create_fake_mask_installation(root: str) -> None:
    if os.path.isdir(root):
        shutil.rmtree(root)

    os.makedirs(root)

    open("{}/mask.npy".format(root)).close()
    
def create_fake_background_installation(root: str) -> None:

