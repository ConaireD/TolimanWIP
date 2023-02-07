"""
P := [E]
E := L, E
  |  L
L := [T]
  |  T
T := N, T
  |  N 

Now consider the example, 
P: [[1, 2], 1, [2, 3, 4]]
E: [1, 2], 1, [2, 3, 4]
L: [1, 2] 
T: 1, 2

"""


def string_to_list(string: str, out: list = []) -> list:
    fragment: str = string[1:-1]
    first_delim: int = fragment.find("[")
    last_delim: int = fragment.rfind("[")

    if first_delim == last_delim:
        entries: list = fragment.split(",")
        inner: list = []
        for entry in entries:
            inner.append(float(entry.strip()))
        out.append(inner)
    else:
        next_delim: int = fragment.find("[", first_delim)
        string_to_list(string[first_delim:next_delim + 1], out)

    return out
