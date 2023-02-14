import os
import sys

def read_lines(file_name: str) -> list:
    with open(file_name, "r") as file:
        lines: list = file.readlines()
    return lines 

def is_code(line: str) -> bool:
    content: str = line.strip()
    pyinput: bool = content.startswith(">>>")
    pyoutput: bool = content.startswith(":::")
    return pyinput or pyoutput

def find_example(lines: str, start: int = 0) -> int:
    lineno: int = start
    length: int = len(lines)
    while not is_code(lines[lineno]):
        lineno: int = lineno + 1
        if lineno >= length:
            return -1
    return lineno

def reformat_examples(lines: str, lineno: int = 0) -> list:
    header: str = "```python"
    lineno: int = find_example(lines, lineno) 
    if lineno < 0:
        return lines
    else:
        if not lines[lineno - 1].find(header) >= 0:
            lines: list = lines[:lineno - 1] + [header] + lines[lineno - 1:]

        lineno: int = lineno + 1
        while is_code(lines[lineno]):
            lineno: int = lineno + 1

        if not lines[lineno].find("```") >= 0:
            lines: list = lines[:lineno] + ["```"] + lines[lineno:]

        return reformat_examples(lines, lineno)

def rewrite_examples(file_name: str) -> None:
    lines: list = read_lines(file_name)
    reformatted: list = reformat_examples(lines)

    with open(file_name, "w") as file:
        file.writelines(reformatted)

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
                rewrite_examples(file)
    elif os.path.isfile(file_name):
        rewrite_examples(file_name)
    else:
        raise ValueError("Please provide a file or directory.")
