"""
P := [E]
E := E, T
  |  T
T := [T]
  |  L 
L := L, N
  |  N
"""
stack: list = []
program: list = []
cursor: int = 0

while cursor < len(program) - 2:
    token: int = program[cursor]
    comeing: int = program[cursor + 1]
    stack: list = stack + [token]

    if stack[-1] == NUMBER:
        stack: list = stack[:-1] + [LIST]
        
        if stack[-2] == COMMA and stack[-3] == LIST:
            stack: list
        cursor: int = cursor - 1
        continue
    
    if stack[-1] == LIST:
        stack: list = stack[:-1] + [EXPRESSION]
        cursor: int = cursor + 1
        continue

    if stack[-1] == 

"""
Now consider the example, 
P: [[1, 2], 1, [2, 3, 4]]
E: [1, 2], 1, [2, 3, 4]
L: [1, 2] 
T: 1, 2
N: 1
T: 2
N: 2
E: 1, [2, 3, 4]
L: 1
T: 1
N: 1
E: [2, 3, 4]
L: 2, 3, 4
N: 2
T: 3, 4
N: 3
T: 4
N: 4
"""

