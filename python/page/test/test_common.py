import unittest
import numpy as np

from page.main import *

class CommonTest(unittest.TestCase):

    def test_parse_model_one_var(self):
        m = ModelExpression.parse("treatment")
        self.assertEquals(m.prob, None)
        self.assertEquals(m.variables, ['treatment'])

    def test_parse_model_two_vars(self):
        m = ModelExpression.parse("treatment + sex")
        self.assertEquals(m.prob, ModelExpression.PROB_MARGINAL)
        self.assertEquals(m.variables, ['treatment', 'sex'])
        
    def test_parse_op(self):
        with self.assertRaises(ModelExpressionException):
            ModelExpression.parse("+")

    def test_parse_var_var(self):
        with self.assertRaises(ModelExpressionException):
            ModelExpression.parse("treatment sex")

    def test_parse_var_op_var_op_var(self):
        with self.assertRaises(ModelExpressionException):
            ModelExpression.parse("treatment + sex + batch")

    def test_str(self):
        self.assertEquals(
            str(ModelExpression.parse("treatment")), "treatment")
        self.assertEquals(
            str(ModelExpression.parse("treatment * sex")), "treatment * sex")


if __name__ == '__main__':
    unittest.main()

