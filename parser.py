LPAREN: int = 0
RPAREN: int = 1
NUMBER: int = 2
COMMA: int = 3
DOT: int = 4

class Token(object):
    token: int
    value: str

    def __init__(self, token: int, value: str = ""):
        self.token = token
        self.value = value

"""
Grammar

P := [E]
E := [E], [E]
  |  [E]
  |  L
L := N, L
  |  N
N := F
  |  I
F := I.I
  | I.
  | .I
I := [0-9]+
"""

def parse_program(tokens: list):
    array: list = []
    if not (tokens[0].token == LPAREN and tokens[-1].token == RPAREN):
        raise ValueError("Mismatched []!")    
    parse_expression(tokens[1:-1], array)
    return array

def parse_expression(tokens: list, array: list) -> list: 
    if tokens[1].token == LPAREN:
        inner: list = parse_expression(tokens[1:], [])
        array.append(inner)
    else:
        parse_terms(tokens[1:-1], array)
        return array

def parse_terms(tokens: list, array: list):
    tokens: list = parse_number(tokens, array)
    if len(tokens) > 1:
        if tokens[0].token == COMMA:
            parse_terms(tokens[1:], array)

def parse_number(tokens: list, array: list):
    if not (tokens[0].token == NUMBER):
        raise ValueError("Not a number!")
    
    if len(tokens) > 1:
        if tokens[1].token == DOT:
            return parse_float(tokens, array)
        else:
            return parse_int(tokens, array)
    return parse_int(tokens, array)

def parse_float(tokens: list, array: list):
    if len(tokens) < 3:
        raise ValueError("Invalid float!")

    if not (tokens[0].token == NUMBER and tokens[1].token == DOT and tokens[2].token == NUMBER):
        raise ValueError("Invalid float")

    number: float = float(tokens[0].value + "." + tokens[2].value)
    array.append(number)
    return tokens[3:]

def parse_int(tokens: list, array: list):
    array.append(int(tokens[0].value))
    return tokens[1:]

def tokenize_program(string: str, tokens: list = []) -> list:
    if len(string) == 0:
        return tokens
    else:
        char: chr = string[0]

        if char == "[":
            string: str = string[1:]
            tokens: list = tokens + [Token(LPAREN)]
            return tokenize_program(string, tokens)
        elif char == "]":
            string: str = string[1:]
            tokens: list = tokens + [Token(RPAREN)]
            return tokenize_program(string, tokens)
        elif char == ",":
            string: str = string[1:] 
            tokens: list = tokens + [Token(COMMA)]
            return tokenize_program(string, tokens)
        elif char == ".":
            string: str = string[1:]
            tokens: list = tokens + [Token(DOT)]
            return tokenize_program(string, tokens)
        elif char.isnumeric():
            number: str = ""
            for digit in string:
                if not digit.isnumeric():
                    break
                number: str = number + digit
            string: str = string[len(number):]
            tokens: list = tokens + [Token(NUMBER, number)]
            return tokenize_program(string, tokens)
        elif char.isspace():
            string: str = string[1:]
            return tokenize_program(string, tokens)
        else:
            raise ValueError("Invalid character!")
        
string: str = "[[1, 2, 3]]"
print(parse_program(tokenize_program(string)))

string: str = "[[1.0, 2.00, 300.0]]"
print(parse_program(tokenize_program(string)))

string: str = "[[1, 2.00, 300.0]]"
print(parse_program(tokenize_program(string)))
