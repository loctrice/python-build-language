##############################################
# IMPORTS
##############################################
from strings_with_arrows import *  # Utility for error visualization.

##############################################
# CONSTANTS
##############################################
DIGITS = '0123456789'  # Characters recognized as part of a number.


##############################################
# ERRORS
##############################################
class Error:
    def __init__(self, pos_start, pos_end, error_name, details):
        self.pos_start = pos_start  # Error starting position.
        self.pos_end = pos_end      # Error ending position.
        self.error_name = error_name  # Type of error.
        self.details = details  # Additional information about the error.

    def as_string(self):
        # Formats the error message for display.
        result = f'{self.error_name}: {self.details}\n'
        result += f'File {self.pos_start.file_name}, line {self.pos_start.line + 1}'
        result += '\n\n' + string_with_arrows(self.pos_start.file_text, self.pos_start, self.pos_end)
        return result


class IllegalCharError(Error):
    # Error for characters that are not recognized by the lexer.
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Illegal Character', details)


class InvalidSyntaxError(Error):
    # Error for syntax that doesn't match the expected patterns.
    def __init__(self,pos_start, pos_end, details=''):
        super().__init__(pos_start, pos_end, 'Invalid Syntax', details)


##############################################
# POSITION
##############################################
class Position:
    def __init__(self, idx, line, col, file_name, file_text):
        # Index in the file, line number, column number, file name, and the file's text.
        self.idx = idx
        self.line = line
        self.column = col
        self.file_name = file_name
        self.file_text = file_text

    def advance(self, current_char=None):
        # Moves to the next position in the file.
        self.idx += 1
        self.column += 1

        if current_char == '\n':  # If newline, move to the next line.
            self.line += 1
            self.column = 0

        return self

    def copy(self):
        return Position(self.idx, self.line, self.column, self.file_name, self.file_text)


##############################################
# TOKENS
##############################################
TT_INT = 'INT'
TT_FLOAT = 'FLOAT'
TT_PLUS = 'PLUS'
TT_MINUS = 'MINUS'
TT_MUL = 'MUL'
TT_DIV = 'DIV'
TT_LPAREN = 'LPAREN'
TT_RPAREN = 'RPAREN'
TT_EOF = 'EOF'  # End-of-file token to indicate the end of input.


class Token:
    def __init__(self, type_, value=None, pos_start=None, pos_end=None):
        self.type = type_  # The type of token.
        self.value = value  # The token's value, if any.

        # Positional information for error reporting.
        if pos_start:
            self.pos_start = pos_start.copy()
            self.pos_end = pos_start.copy()
            self.pos_end.advance(None)
        if pos_end:
            self.pos_end = pos_end.copy()

    def __repr__(self):
        # Representation of the token for debugging.
        if self.value: return f'{self.type}:{self.value}'
        return f'{self.type}'


##############################################
# LEXER
##############################################
class Lexer:
    def __init__(self, file_name, text):
        self.file_name = file_name  # Name of the file being interpreted.
        self.text = text  # Text content of the file.
        self.pos = Position(-1, 0, -1, file_name, text)  # Position in the text.
        self.current_char = None  # Current character being analyzed.
        self.advance()  # Initialize the first character.

    def advance(self):
        # Move to the next character in the text.
        self.pos.advance(self.current_char)
        if self.pos.idx < len(self.text):
            self.current_char = self.text[self.pos.idx]
        else:
            self.current_char = None  # End of file.

    def make_tokens(self):
        tokens = []  # List of tokens generated from the text.

        while self.current_char is not None:
            # Skip whitespace characters.
            if self.current_char in ' \t':
                self.advance()
            # Skip delimiters like ';' and newlines for now.
            elif self.current_char in ';\n':
                self.advance()
            # Number detection.
            elif self.current_char in DIGITS:
                tokens.append(self.make_number())
            # Operator detection.
            elif self.current_char == '+':
                tokens.append(Token(TT_PLUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == '-':
                tokens.append(Token(TT_MINUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == '*':
                tokens.append(Token(TT_MUL, pos_start=self.pos))
                self.advance()
            elif self.current_char == '/':
                tokens.append(Token(TT_DIV, pos_start=self.pos))
                self.advance()
            # Parentheses detection.
            elif self.current_char == '(':
                tokens.append(Token(TT_LPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == ')':
                tokens.append(Token(TT_RPAREN, pos_start=self.pos))
                self.advance()
            # Unknown character error.
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.advance()
                return [], IllegalCharError(pos_start, self.pos, "'" + char + "'")

        tokens.append(Token(TT_EOF, pos_start=self.pos))  # End-of-file token.
        return tokens, None  # Return the list of tokens and no error.

    def make_number(self):
        # Parses a number (integer or float).
        num_str = ''
        dot_count = 0  # Count of decimal points in the number.
        pos_start = self.pos.copy()

        while self.current_char is not None and self.current_char in DIGITS + '.':
            if self.current_char == '.':
                if dot_count == 1: break  # Only one decimal point is allowed.
                dot_count += 1
                num_str += '.'
            else:
                num_str += self.current_char
            self.advance()

        if dot_count == 0:
            return Token(TT_INT, int(num_str), pos_start, self.pos)  # Integer token.
        else:
            return Token(TT_FLOAT, float(num_str), pos_start, self.pos)  # Float token.


##############################################
# NODES
##############################################
class NumberNode:
    def __init__(self, tok):
        self.tok = tok  # The token representing the number.
        self.pos_start = self.tok.pos_start  # Start position of the number in the text.
        self.pos_end = self.tok.pos_end  # End position of the number in the text.

    def __repr__(self):
        return f'{self.tok}'


class BinOpNode:
    def __init__(self, left_node, operator_token, right_node):
        # Binary operation node (e.g., addition, subtraction).
        self.left_node = left_node  # The left operand.
        self.operator_token = operator_token  # The operator token (e.g., PLUS, MINUS).
        self.right_node = right_node  # The right operand.

        # Start and end positions cover the entire expression.
        self.pos_start = self.left_node.pos_start
        self.pos_end = self.right_node.pos_end

    def __repr__(self):
        return f'({self.left_node}, {self.operator_token}, {self.right_node})'


class UnaryOpNode:
    def __init__(self, operator_token, node):
        # Unary operation node (e.g., negation).
        self.operator_token = operator_token  # The operator token (e.g., MINUS).
        self.node = node  # The operand.

        # Start and end positions are determined by the operator and operand.
        self.pos_start = self.operator_token.pos_start
        self.pos_end = node.pos_end

    def __repr__(self):
        return f'({self.operator_token}, {self.node})'


##############################################
# PARSE RESULT
##############################################
class ParseResult:
    # Represents the result of a parse operation.
    def __init__(self):
        self.error = None  # Any error that occurred during parsing.
        self.node = None  # The root node of the parsed AST.

    def register(self, res):
        # Registers another parse result or node, capturing errors.
        if isinstance(res, ParseResult):
            if res.error: self.error = res.error
            return res.node
        return res

    def success(self, node):
        # Marks this parse result as successful.
        self.node = node
        return self

    def failure(self, error):
        # Marks this parse result as a failure.
        self.error = error
        return self


##############################################
# PARSER
##############################################
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens  # List of tokens to be parsed.
        self.tok_idx = -1  # Current index in the tokens list.
        self.current_tok = None  # Current token being parsed.
        self.advance()  # Initialize the first token.

    def advance(self):
        # Moves to the next token in the list.
        self.tok_idx += 1
        if self.tok_idx < len(self.tokens):
            self.current_tok = self.tokens[self.tok_idx]
        return self.current_tok

    def parse(self):
        # The entry point for parsing.
        res = self.expr()
        if not res.error and self.current_tok.type != TT_EOF:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '+', '-', '*' or '/'"
            ))
        return res

    def factor(self):
        # Parses a factor (numbers, parentheses expressions, unary operations).
        res = ParseResult()
        tok = self.current_tok

        if tok.type in (TT_PLUS, TT_MINUS):
            # Unary operation.
            # This condition checks if the current token is a unary plus or minus sign.
            # Unary operations apply to a single operand. For example, -5 or +3.

            res.register(self.advance())
            # Advance to the next token after the unary operator.
            # This is because the unary operator applies to the following factor,
            # so we need to move forward in the token stream to parse that factor.

            factor = res.register(self.factor())
            # Recursively call the `factor` method to parse the operand of the unary operation.
            # This operand can be a simple number, another expression within parentheses,
            # or even another unary operation. The recursion allows handling nested structures.

            if res.error: return res
            # If an error occurred during the parsing of the factor (e.g., invalid syntax),
            # return immediately with the error. This halts further parsing.

            return res.success(UnaryOpNode(tok, factor))
            # If parsing the factor succeeds, wrap the parsed factor and the unary operator
            # in a `UnaryOpNode` (a node representing a unary operation in the AST) and
            # mark the parsing operation as successful by calling `res.success`.
            # This `UnaryOpNode` will then be part of the AST, representing the unary operation.

        elif tok.type in (TT_INT, TT_FLOAT):
            # Number.
            # This condition checks if the current token is an integer or a float.
            # These types are direct values and do not need further parsing.

            res.register(self.advance())
            # Advance to the next token since we've recognized this token as a complete factor.

            return res.success(NumberNode(tok))
            # Create a `NumberNode` with the current token to represent it in the AST,
            # then mark this operation as successful. This node will directly hold the
            # numeric value found in the source code.
        elif tok.type == TT_LPAREN:
            # Parentheses expression.
            # This condition checks if the current token is a left parenthesis, indicating
            # the start of an expression within parentheses.

            res.register(self.advance())
            # Advance to the next token to start parsing the expression inside the parentheses.

            expr = res.register(self.expr())
            # Recursively call the `expr` method to parse the expression inside the parentheses.
            # This can be any valid expression recognized by the parser.

            if res.error: return res
            # If an error occurred during the parsing of the expression, return immediately
            # with the error. This stops further parsing and propagates the error up.

            if self.current_tok.type == TT_RPAREN:
                # Check if the current token is a right parenthesis, which would correctly
                # close the expression.

                res.register(self.advance())
                # Advance to the next token after the right parenthesis, having successfully
                # parsed the parentheses-enclosed expression.

                return res.success(expr)
                # Mark the parsing operation as successful and return the parsed expression.
                # The expression itself becomes a node in the AST, representing the entire
                # parentheses-enclosed part.

        # If the token does not match any expected factor types (number or parentheses-enclosed expression).
        return res.failure(InvalidSyntaxError(
            tok.pos_start, tok.pos_end, "Expected int or float"
        ))

    def term(self):
        # Parses a term (factors separated by '*' or '/').
        return self.bin_op(self.factor, (TT_MUL, TT_DIV))

    def expr(self):
        # Parses an expression (terms separated by '+' or '-').
        return self.bin_op(self.term, (TT_PLUS, TT_MINUS))

    def bin_op(self, func, ops):
        # Generic method for parsing binary operations.
        # func: A function that parses the operands of the binary operation.
        # ops: A tuple containing the token types that this method should treat as binary operators.

        res = ParseResult()

        left = res.register(func())
        # Parse the left operand of the binary operation by calling the provided function `func`.
        # This could be another call to `expr`, `term`, or `factor` method, depending on the context.
        # The result is registered with the current parse result to capture any errors or the parsed node.

        if res.error: return res

        while self.current_tok.type in ops:
            # Loop as long as the current token is one of the operators specified in `ops`.
            # This allows handling expressions like "1 + 2 + 3" or "4 * 5 / 6" correctly,
            # parsing them as a series of binary operations.

            operator_token = self.current_tok
            # Store the current token as the operator of the binary operation.

            res.register(self.advance())
            # Advance to the next token to parse the right operand of the binary operation.

            right = res.register(func())
            # Parse the right operand of the binary operation, similarly to how the left operand was parsed.

            if res.error: return res
            # If there was an error parsing the right operand, return immediately with the error.

            left = BinOpNode(left, operator_token, right)
            # Create a new `BinOpNode` representing this binary operation.
            # The current `left` node becomes the left operand, the `operator_token` is the operator,
            # and the newly parsed `right` node is the right operand.
            # This `BinOpNode` then becomes the new `left` node for the next iteration of the loop,
            # allowing the method to handle chains of binary operations.

        return res.success(left)
        # Once all binary operations have been parsed, return a success result with the final `left` node,
        # which now represents the entire chain of binary operations as a single AST node.


##############################################
# NUMBER
##############################################
class Number:
    def __init__(self, value):
        self.value = value  # The numeric value.
        self.set_pos()

    def set_pos(self, pos_start=None, pos_end=None):
        # Sets the position of this number in the source code.
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def added_to(self, other):
        # Adds this number to another Number object.
        if isinstance(other, Number):
            return Number(self.value + other.value)

    def subbed_by(self, other):
        # Subtracts another Number object from this one.
        if isinstance(other, Number):
            return Number(self.value - other.value)

    def multiplied_by(self, other):
        # Multiplies this number by another Number object.
        if isinstance(other, Number):
            return Number(self.value * other.value)

    def divided_by(self, other):
        # Divides this number by another Number object.
        if isinstance(other, Number):
            if other.value == 0: return None  # Avoid division by zero.
            return Number(self.value / other.value)

    def __repr__(self):
        # Representation of the number for debugging.
        return f'{self.value}'


##############################################
# INTERPRETER
##############################################
class Interpreter:
    def visit(self, node):
        # Dispatches node to the appropriate visit method.
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.no_visit_method)
        return method(node)

    def no_visit_method(self, node):
        # Called if no visit method is defined for a node type.
        raise Exception(f'No visit_{type(node).__name__} method defined')

    def visit_NumberNode(self, node):
        # Visits a NumberNode to produce a Number object.
        return Number(node.tok.value).set_pos(node.pos_start, node.pos_end)

    def visit_BinOpNode(self, node):
        # Visits a BinOpNode to perform the binary operation.
        left = self.visit(node.left_node)
        right = self.visit(node.right_node)

        if node.operator_token.type == TT_PLUS:
            result = left.added_to(right)
        elif node.operator_token.type == TT_MINUS:
            result = left.subbed_by(right)
        elif node.operator_token.type == TT_MUL:
            result = left.multiplied_by(right)
        elif node.operator_token.type == TT_DIV:
            result = left.divided_by(right)

        return result.set_pos(node.pos_start, node.pos_end)

    def visit_UnaryOpNode(self, node):
        # Visits a UnaryOpNode to perform the unary operation.
        number = self.visit(node.node)

        if node.operator_token.type == TT_MINUS:
            number.value = -number.value

        return number.set_pos(node.pos_start, node.pos_end)


##############################################
# RUN
##############################################
def run(file_name, text):
    lexer = Lexer(file_name, text)  # Lexical analysis.
    tokens, error = lexer.make_tokens()
    if error: return None, error  # Return early on lexical error.

    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error: return None, ast.error  # Return early on syntax error.

    interpreter = Interpreter()  # Interpretation.
    #print the ast
    print(ast.node)

    result = interpreter.visit(ast.node)

    return result, ast.error
