LPAREN: int = 0
RPAREN: int = 1
NUMBER: int = 2
COMMA: int = 3

# So what do we have?
#
# P := E
# E := [E]
#   | [L, E]
# L := [T] 
# T := N, T
#   | N

# N := D.D
#   | D
# D := [0-9]+

def parse_program(tokens: list):
    parse_expression(tokens)
    print("\n", end="")

def parse_expression(tokens: list): 
    if not (tokens[0] == LPAREN and tokens[-1] == RPAREN):
        raise ValueError("Mismatched []!")
    
    print("[", end="")
    if tokens[1] == LPAREN and tokens[-2] == RPAREN:
        parse_expression(tokens[1:-1])
    else:
        parse_terms(tokens[1:-1])
    print("]", end="")

def parse_terms(tokens: list):
    if not (tokens[0] == NUMBER):
        raise ValueError("Not a number!")

    parse_number(tokens[0])

    if len(tokens) > 1:
        if tokens[1] == COMMA:
            print(",", end="")
            parse_terms(tokens[2:])

def parse_number(tokens: list):
    print("NUMBER", end="")

def tokenize(string: str) -> iter:
    stream: list = list()
    for char in string:
        if char == "[":
            stream.append(LPAREN)
        elif char == "]":
            stream.append(RPAREN)
        elif char == ",":
            stream.append(COMMA)
        elif char.isnumeric():
            while char.inumeric()
            stream.append(NUMBER)
        elif char.isspace():
            continue
        else:
            raise ValueError
    return stream

string: str = "[[1, 2, 3]]"
parse_program(tokenize(string))

