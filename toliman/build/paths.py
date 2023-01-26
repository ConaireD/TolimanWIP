import os 

def accumulate(strings: list, paths: str = []) -> list:
    """
    Incrementally build a path from a list.

    Parameters
    ----------
    strings: list 
        A list of directories that can be pasted together to form a 
        list.
    paths: list
        A list of the paths to each dir.

    Returns
    -------
    paths: list
        Starting with the root directory, which is assumed to be the 
        first entry in `strings`, a list of directories growing from
        root.

    Examples
    --------
    >>> strings: list = ["root", "dev", "null"]
    >>> _accumulate_path(strings)
    ::: ["root", "root/dev", "root/dev/null"]
    """
    if not strings:
        return paths
    else:
        if not paths:
            paths.append(strings.pop(0))
        else:
            paths.append(paths[-1] + "/" + strings.pop(0))
        return _accumulate_path(strings, paths)
