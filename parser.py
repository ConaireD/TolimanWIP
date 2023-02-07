LPAREN: int = 0
RPAREN: int = 1
NUMBER: int = 2
COMMA: int = 3
DOT: int = 4

# So what do we have?
#
# P := E
# E := [E]
#   | [L, E]
# L := [T] 
# T := N, T
#   | N
# N := F
#   | I
# N := I.I

def parse_program(tokens: list):
    parse_expression(tokens)
    print("\n", end="")

def parse_expression(tokens: list): 
    # print("E: ", tokens)
    if not (tokens[0] == LPAREN and tokens[-1] == RPAREN):
        raise ValueError("Mismatched []!")
    
    print("[", end="")
    if tokens[1] == LPAREN and tokens[-2] == RPAREN:
        parse_expression(tokens[1:-1])
    else:
        parse_terms(tokens[1:-1])
    print("]", end="")

def parse_terms(tokens: list):
    # print("T: ", tokens)
    tokens: list = parse_number(tokens)
    if len(tokens) > 1:
        if tokens[0] == COMMA:
            print(",", end="")
            parse_terms(tokens[1:])

def parse_number(tokens: list):
    # print("N: ", tokens)
    if not (tokens[0] == NUMBER):
        raise ValueError("Not a number!")
    
    if len(tokens) > 1:
        if tokens[1] == DOT:
            return parse_float(tokens)
        else:
            return parse_int(tokens)
    return parse_int(tokens)

def parse_float(tokens: list):
    # print("F: ", tokens)
    if len(tokens) < 3:
        raise ValueError("Invalid float!")

    if not (tokens[0] == NUMBER and tokens[1] == DOT and tokens[2] == NUMBER):
        raise ValueError("Invalid float")

    print("FLOAT", end="")
    return tokens[3:]

def parse_int(tokens: list):
    # print("I: ", tokens)
    print("INT", end="")
    return tokens[1:]

def tokenize_number(string: str) -> int:
    if len(string) == 0:
        return string
    elif string[0].isnumeric():
        return tokenize_number(string[1:])
    else:
        return string

def tokenize_program(string: str, tokens: list = []) -> list:
    if len(string) == 0:
        return tokens
    else:
        char: chr = string[0]

        if char == "[":
            string: str = string[1:]
            tokens: list = tokens + [LPAREN]
            return tokenize_program(string, tokens)
        elif char == "]":
            string: str = string[1:]
            tokens: list = tokens + [RPAREN]
            return tokenize_program(string, tokens)
        elif char == ",":
            string: str = string[1:] 
            tokens: list = tokens + [COMMA]
            return tokenize_program(string, tokens)
        elif char == ".":
            string: str = string[1:]
            tokens: list = tokens + [DOT]
            return tokenize_program(string, tokens)
        elif char.isnumeric():
            string: str = tokenize_number(string)
            tokens: list = tokens + [NUMBER]
            return tokenize_program(string, tokens)
        elif char.isspace():
            string: str = string[1:]
            return tokenize_program(string, tokens)
        else:
            raise ValueError("Invalid character!")
        
string: str = "[[1, 2, 3]]"
print(tokenize_program(string))
parse_program(tokenize_program(string))

string: str = "[[1.0, 2.00, 300.0]]"
print(tokenize_program(string))
parse_program(tokenize_program(string))

string: str = "[[1, 2.00, 300.0]]"
print(tokenize_program(string))
parse_program(tokenize_program(string))
