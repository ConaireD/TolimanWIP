LPAREN: int = 0
RPAREN: int = 1
NUMBER: int = 2
COMMA: int = 3
DOT: int = 4

class Token(object):
    """
    Represents a character or group of characters in the input stream.

    The token is an abstraction allowing the syntax to be separated from
    the information and parsed. 

    Attributes
    ----------
    token: int
        The tokens are encoded by an enum with the following key.
        LPAREN = 0, RPAREN = 1, NUMBER = 2, COMMA = 3 and DOT = 4.
    value: str, optional
        The information. This is only included for the NUMBER token.
        Because (unfortunately) this is not C, it may have made more 
        sense to inherit Number from Token and include the value 
        field specifically in Number. This also works so I have not.
    """
    token: int
    value: str

    def __init__(self, token: int, value: str = ""):
        """
        Parameters
        ----------
        token: int 
            The syntax.
        value: str = ""
            The information.
        """
        self.token = token
        self.value = value

def parse_program(tokens: list) -> list:
    """
    Parse an arbitrarily nested 1d array with mixed types.

    Parameters
    ----------
    tokens: list
        A list of tokens representing the program.

    Returns
    -------
    array: list
        A list of mixed types representing the ouput of the program.
    """
    array: list = []
    parse_expression(tokens, array)
    return array

def parse_expression(tokens: list, array: list) -> list: 
    """
    The grammar of an expression is:

    E := [E]
      |  [T]

    Parameters
    ----------
    tokens: list
        The tokens representing the recursive fragement of the program.
    array: list
        
    """
    if not (tokens[0].token == LPAREN and tokens[-1].token == RPAREN):
        raise ValueError("Mismatched []!")    

    if tokens[1].token == LPAREN and tokens[-2].token == RPAREN:
        inner: list = parse_expression(tokens[1:-1], [])
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
