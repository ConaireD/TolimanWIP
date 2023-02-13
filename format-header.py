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

    inline: bool = line.split("[")[-1].strip()
    
    if len(inline) == 0:
        api: list = []
        line: str = next(lines_iter)
        while not line.strip().startswith("]"):
            api: list = api + [line.strip().strip(",")]
            line: str = next(lines_iter)
    else:
        api: list = [line.strip() for line in inline.strip("]").split(",")]

    print(api)
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Please provide the name of the file to document.")

    file_name: str = sys.argv[1]

    if os.path.isdir(file_name):
        for item in os.walk(file_name):
            dirname: str = item[0]
            fnames: list = item[2]
            files: list = [dirname + "/" + fname for fname in fnames]
            for file in files:
                create_header(file)
    elif os.path.isfile(file_name):
        create_header(file_name)
    else:
        raise ValueError("Please provide a file or directory.")
