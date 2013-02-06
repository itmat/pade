import numpy as np
from collections import namedtuple
from itertools import combinations, product
import StringIO
import tokenize
import logging

class ModelExpressionException(Exception):
    """Thrown when a model expression is invalid."""
    pass


DummyVarTable = namedtuple(
    "DummyVarTable",
    ["names", "rows"])

DummyVarAssignment = namedtuple(
    "DummyVarAssignment",
    ["factor_values",
     "bits",
     "indexes"])

FittedModel = namedtuple(
    "FittedModel",
    ["labels",
     "x",
     "y_indexes",
     "params"])


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
        self.validate_model()

    def validate_model(self):
        """Validate the model against the given schema.

        Raises an exception if the model refers to any variables that are
        not defined in schema.
        
        """
        
        for factor in self.expr.variables:
            if factor not in self.schema.factors:
                raise Exception("Factor '" + factor + "' is not defined in the schema. Valid factors are " + str(self.schema.factors.keys()))

    @property
    def layout(self):
        s = self.schema
        return [s.indexes_with_assignments(a)
                for a in s.possible_assignments(self.expr.variables)]

    def fit(self, data):

        logging.info("Computing coefficients using least squares for " +
                 str(len(data)) + " rows")

        effect_level = 1 if self.expr.operator == '+' else len(self.expr.variables)

        dummies = dummy_vars(self.schema, level=effect_level)

        x = []
        indexes = []

        for row in dummies.rows:
            for index in row.indexes:
                x.append(row.bits)
                indexes.append(self.schema.sample_name_index[index])

        x = np.array(x, bool)

        num_vars = np.size(x, axis=1)
        shape = np.array((len(data), num_vars))

        result = np.zeros(shape)

        #for i, row in enumerate(data):
        #    y = row[indexes]
        #    (coeffs, residuals, rank, s) = np.linalg.lstsq(x, y)
        #    result[i] = coeffs

        return FittedModel(dummies.names, x, indexes, result)


def dummy_vars(schema, factors=None, level=None):
    """
    level=0 is intercept only
    level=1 is intercept plus main effects
    level=2 is intercept, main effects, interactions between two variables
    ...
    level=n is intercept, main effects, interactions between n variables

    """ 
    factors = schema._check_factors(factors)

    if level is None:
        return dummy_vars(schema, factors, len(factors))

    if level == 0:
        names = ({},)
        rows = []
        for a in schema.possible_assignments(factors):
            rows.append(
                DummyVarAssignment(a.values(), (True,), schema.samples_with_assignments(a)))
        return DummyVarTable(names, rows)

    res = dummy_vars(schema, factors, level - 1)

    col_names = tuple(res.names)
    rows      = list(res.rows)

    # Get col names
    for interacting in combinations(factors, level):
        for a in schema.factor_combinations(interacting):
            if schema.has_baseline(dict(zip(interacting, a))):
                continue
            col_names += ({ interacting[i] : a[i] for i in range(len(interacting)) },)

    for i, dummy in enumerate(rows):
        (factor_values, bits, indexes) = dummy

        # For each subset of factors of size level
        for interacting in combinations(factors, level):

            my_vals = ()
            for j in range(len(factors)):
                if factors[j] in interacting:
                    my_vals += (factor_values[j],)

            # For each possible assignment of values to these factors
            for a in schema.factor_combinations(interacting):
                if schema.has_baseline(dict(zip(interacting, a))):
                    continue

                # Test if this row of the result table has all the
                # values in this assignment
                matches = my_vals == a
                bits = bits + (matches,)

        rows[i] = DummyVarAssignment(tuple(factor_values), bits, indexes)

    return DummyVarTable(col_names, rows)


