import numpy as np
from collections import namedtuple
import StringIO
import tokenize
import logging

class ModelExpressionException(Exception):
    """Thrown when a model expression is invalid."""
    pass


class ModelExpression:
    
    def __init__(self, 
                 operator=None,
                 variables=None):

        if variables is None:
            variables = []

        # If we have two or mor vars, make sure we have a valid operator
        if len(variables) > 1:
            if operator is None:
                raise Exception("Operator must be supplied for two or more variables")
            if operator not in "+*":
                raise Exception("Operator must be '+' or '*'")

        self.operator = operator
        self.variables = variables

        
    @classmethod
    def parse(cls, string):
        """Parse a model from a string.

        string can either be "VARIABLE", "VARIABLE * VARIABLE", or
        "VARIABLE + VARIABLE". We may support more complex models
        later on.

        """

        if string is None or string.strip() == '':
            return ModelExpression()

        operator = None
        variables = []

        io = StringIO.StringIO(string)
        toks = tokenize.generate_tokens(lambda : io.readline())

        # First token should always be a variable
        t = toks.next()
        (tok_type, tok, start, end, line) = t
        if tok_type != tokenize.NAME:
            raise ModelExpressionException("Unexpected token " + tok)
        variables.append(tok)

        # Second token should be either the end marker or + or *
        t = toks.next()
        (tok_type, tok, start, end, line) = t
        if tok_type == tokenize.ENDMARKER:
            return ModelExpression(variables=variables)
        elif tok_type == tokenize.OP:
            operator = tok
        else:
            raise ModelExpressionException("Unexpected token " + tok)

        t = toks.next()
        (tok_type, tok, start, end, line) = t
        if tok_type != tokenize.NAME:
            raise ModelExpressionException("Unexpected token " + tok)
        variables.append(tok)

        t = toks.next()
        (tok_type, tok, start, end, line) = t
        if tok_type != tokenize.ENDMARKER:
            raise ModelExpressionException("Expected end of expression, got " + tok)

        if len(variables) > 0:
            return ModelExpression(operator=operator, variables=variables)

    def __str__(self):
        if len(self.variables) == 0:
            return ""
        elif len(self.variables) == 1:
            return self.variables[0]
        else:
            op = " " + self.operator + " "
            return op.join(self.variables)


class Model:
    def __init__(self, schema, expr):
        self.schema = schema
        self.expr = ModelExpression.parse(expr)
        for factor in self.expr.variables:
            if factor not in self.schema.factors:
                raise Exception(
                    "Factor '" + factor + 
                    "' is not defined in the schema. Valid factors are " + 
                    str(self.schema.factors.keys()))

    @property
    def layout(self):
        s = self.schema
        return [s.indexes_with_assignments(a)
                for a in s.possible_assignments(self.expr.variables)]


