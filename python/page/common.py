import tokenize
from StringIO import StringIO

class ModelExpressionException(Exception):
    pass

class Model:
    
    PROB_MARGINAL    = "marginal"
    PROB_JOINT       = "joint"
    PROB_CONDITIOANL = "conditional"

    


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
            if   tok == '+': operator = cls.PROB_MARGINAL
            elif tok == '*': operator = cls.PROB_JOINT
            else:
                raise ModelExpressionException("Unexpected operator " + tok)
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


