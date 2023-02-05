import os 

__author__ = "Jordan Dennis"
__all__ = [
    "accumulate",
    "concat",
]

def accumulate(strings: list) -> list:
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
    def accumulate(strings: list, paths: list) -> list:
        if not strings:
            return paths
        else:
            if not paths:
                paths.append(strings.pop(0))
            else:
                paths.append(concat([paths[-1], strings.pop(0)]))
            return accumulate(strings, paths)

    paths: list = []
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

def mkdir_and_parents(root: str) -> None:
    """
    Ensure the root directory exists.

    Parameters
    ----------
    root: str
        The directory to search for an installation in.

    Examples
    --------
    >>> import os
    >>> mkdir_and_parents(".assets")
    >>> os.path.isdir(".assets")
    ::: True
    >>> mkdir_and_parents(".assets/mask/raw")
    >>> os.path.isdir(".assets/mask/raw")
    ::: True
    """
    if not os.path.exists(root):
        for path in accumulate(root.split("/")):
            if not os.path.exists(path):
                os.mkdir(path)

