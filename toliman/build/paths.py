import os 

__author__ = "Jordan Dennis"
__all__ = [
    "accumulate",
    "concat",
]


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
    >>> _accumulate_path(["root", "dev", "null"])
    ::: ["root", "root/dev", "root/dev/null"]
    """
    if not strings:
        return paths
    else:
        if not paths:
            paths.append(strings.pop(0))
        else:
            paths.append(concat([paths[-1], strings.pop(0)]))
        return accumulate(strings, paths)

def concat(paths: list) -> str:
    """
    Fuse paths together.

    Parameters
    ----------
    paths: list
        A list of paths/files to concatenate.

    Returns
    -------
    path: str
        A path made from the other paths.

    Examples
    --------
    >>> concat(["root", "dev", "null"])
    ::: "root/dev/null"`
    """
    return "/".join(paths)
