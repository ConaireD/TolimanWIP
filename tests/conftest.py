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

def all_true(bools: list, true: bool = True) -> bool:
    if not true:
        return False
    elif not bools:
        return True
    else:
        for elem in bools:
            if isinstance(elem, list):
                return true and _all_true(elem, true)
            else:
                return true and bools.pop() and _all_true(bools)

def is_tensor(tensor: list) -> bool:
    def __is_tensor(tensor: list, shape: int) -> bool:
        nests: list = [isinstance(elem, list) for elem in tensor]
        if not all_true(nests):
            for nest in nests:
                if nest:
                    return False
            return len(tensor) == shape
        else:
            true: bool = True
            length: int = len(tensor[0])
            for nested in tensor:
                true: bool = true and _is_tensor(nested, length)

            return true
    if isinstance(tensor[0], list):
        return __is_tensor(tensor, len(tensor[0]))
    else: 
        for elem in tensor:
            if isinstance(elem, list):
                return False
        return True

def tensor_nesting(tensor: list) -> list: 
    def _tensor_nesting(tensor: list, level: int = 0) -> int:
        if not isinstance(tensor[0], list):
            return level 
        else:
            _tensor_nesting(tensor[0], level + 1)

    if not is_tensor(tensor):
        return ValueError
    else:
        return _tensor_nesting(tensor)
            
def is_flat_tensor(tensor: list) -> list:
    nests: list = [isinstance(elem, list) for elem in tensor]
    for nest in nests:
        if nest:
            return False
    return True

def print_tensor(tensor: list) -> None:
    def _print_tensor(tensor: list, nesting: int = 0) -> None:
        if not isinstance(tensor, list):
            print(nesting * "  " + "{},".format(tensor))
        elif is_flat_tensor(tensor): 
            print(nesting * "  " + "{},".format(tensor))
        else:
            print(nesting * "  " + "[")
            for elem in tensor:
                _print_tensor(elem, nesting + 1)
            print(nesting * "  " + "]")

    return _print_tensor(tensor)

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
