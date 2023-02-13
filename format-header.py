import os
import sys 

def create_header(file_name: str) -> None:
    with open(file_name) as file:
        lines: list = file.readlines()

    try:
        lines_iter: iter = iter(lines)
        line: str = next(lines_iter)
        while not line.startswith("__all__"):
            line: str = next(lines_iter).strip()
    except StopIteration as stop_iter:
        return

    inline: bool = line.split("[")[-1].strip().isempty()
    print("inline: [{}]".format(inline))


    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Please provide the name of the file to document.")

    file_name: str = sys.argv[1]
    create_header(file_name)
