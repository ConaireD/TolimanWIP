import os
import sys 

def read_lines(file_name: str) -> list:
    with open(file_name) as file:
        lines: list = file.readlines()
    return lines

def harvest_api(lines: list) -> list:
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
            api: list = api + [line.strip().strip(",").strip('"')]
            line: str = next(lines_iter)
    else:
        api: list = [line.strip() for line in inline.strip("]").split(",")]

    return api

def get_mkdocs_path(file_name: str) -> str:
    return ".".join(file_name.strip(".py").split("/"))

def write_api_header(file_name: str) -> None:
    lines: list = read_lines(file_name)
    apis: list = harvest_api(lines)
    mkdocs_path: str = get_mkdocs_path(file_name)

    if not apis:
        return

    with open(file_name, "w") as file:
        file.write('"""md' + os.linesep)
        file.write("## API" + os.linesep)
        for api in apis:
            file.write("??? note `{}`".format(api) + os.linesep)
            file.write("    ::: {}.{}".format(mkdocs_path, api) + os.linesep * 2)
        file.write('"""' + os.linesep * 2)
        file.writelines(lines)

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
                write_api_header(file)
    elif os.path.isfile(file_name):
        write_api_header(file_name)
    else:
        raise ValueError("Please provide a file or directory.")
