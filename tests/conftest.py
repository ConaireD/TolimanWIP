import os 

ASSETS: str = "toliman/assets"
PHOENIX: str = "{}/grid/phoenix".format(ASSETS)
PHOENIXS: str = ["phoenixm00", "phoenixp03"]
NUMBERS: list = [5200, 5300, 5700, 5800]
PHOENIXS: list = [
    "{}/{}/{}_{}.fits".format(PHOENIX, phoe, phoe, num) 
        for num in NUMBERS 
        for phoe in PHOENIXS
]

def _all_true(bools: list, true: bool = True) -> bool:
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

def _is_tensor(tensor: list) -> bool:
    def __is_tensor(tensor: list, shape: int) -> bool:
        nests: list = [isinstance(elem, list) for elem in tensor]
        if not _all_true(nests):
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

    if not _is_tensor(tensor):
        return ValueError
    else:
        return _tensor_nesting(tensor)
            
simple: list = [[1, 1, 1, 1], [1, 1, 1, 1]]
simple2: list = [[[1, 1], [1, 1], [1, 1]]]
comp: list = [[1, 1], [1, 1, 1], [1, [1, 1]]]

def print_tensor(tensor: list) -> None:
    max_nesting: int = tensor_nesting(tensor)
    
    def _print_tensor(tensor: list, nesting: int = 0) -> None:
        if nesting == max_nesting:
            print(nesting * "  " + "{},".format(tensor))
        else:
            print(nesting * "  " + "[")
            for elem in tensor:
                _print_tensor(elem, nesting + 1)
            print(nesting * "  " + "]")

    return _print_tensor(tensor)


def setup_assets():
    pass

def _teardown_assets():
    pass
