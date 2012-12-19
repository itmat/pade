import tokenize
import numpy as np
import os
import contextlib
from StringIO import StringIO

class ModelExpressionException(Exception):
    pass


class Model:
    
    PROB_MARGINAL    = "marginal"
    PROB_JOINT       = "joint"

    OP_TO_NAME = { '+' : PROB_MARGINAL, '*' : PROB_JOINT }
    NAME_TO_OP = { PROB_MARGINAL : '+', PROB_JOINT : '*' }

    def __init__(self, prob=None, variables=None):
        self.prob      = prob
        self.variables = variables
        
    @classmethod
    def parse(cls, string):
        """Parse a model from a string.

        string can either be "VARIABLE", "VARIABLE * VARIABLE", or
        "VARIABLE + VARIABLE". We may support more complex models
        later on.

        """
        operator = None
        variables = []

        io = StringIO(string)
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
            return Model(variables=variables)
        elif tok_type == tokenize.OP:
            operator = Model.OP_TO_NAME[tok]
            # raise ModelExpressionException("Unexpected operator " + tok)
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

        return Model(prob=operator, variables=variables)

    def __str__(self):
        if len(self.variables) == 1:
            return self.variables[0]
        else:
            op = ' ' + self.NAME_TO_OP[self.prob] + ' '
            return op.join(self.variables)

def add_condition_axis(data, layout):

    num_samples = len(data)
    num_conditions = len(layout)
    num_samples_per_condition = len(layout[0])

    if num_conditions * num_samples_per_condition != num_samples:
        raise Exception("Bad layout")

    shape = (num_conditions, num_samples_per_condition) + np.shape(data)[1:]
    res = np.zeros(shape)

    for i, cond_cols in enumerate(layout):
        res[i] = data[layout[i]]

    return res

@contextlib.contextmanager
def chdir(path):
    cwd = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(cwd)
    
