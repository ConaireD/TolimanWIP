"""md
## Overview
`toliman` deals with a large number of data files, that need to be cast 
into arrays. Some of these data files are `csv`s. Rather than add an 
additional dependency like pandas, I wrote my own csv parser, designed 
to be relatively simple. I tried to use `jax` as much as possible. 
This was a stupid idea since the function cannot be compiled, since it 
contains side effects. A slightly faster and more concise version of this
function can be achieved using a list comprehension or the inbuilt `map`
function. Regardless if it ain't broke don't fix it.

## API
??? note "`read_csv_to_jax_array`"
    ::: toliman.io.read_csv_to_jax_array
"""

import jax 
import jax.numpy as np

__author__ = ["Jordan Dennis"]
__all__ = ["read_csv_to_jax_array"]

strip: callable = lambda _str: _str.strip().split(",")
str_to_float: callable = lambda _str: float(_str.strip())

def read_csv_to_jax_array(file_name: str) -> float:
    """
    Read a CSV using `jax`.

    This is a private function and following convention it assumes that the 
    file exists. There is no error checking!

    Parameters
    ----------
    _file_name: str
        The name of the file to read.

    Returns
    -------
    arr: float
        The information in the CSV. The headings are not returned and so it 
        is up to you to keep track of what each column is.
    """
    with open(file_name, "r") as file:
        lines: list = file.readlines()
        _: str = lines.pop(0)
        entries: list = jax.tree_map(strip, lines)
        file: float = jax.tree_map(str_to_float, entries)
    return np.array(file)
