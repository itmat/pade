import unittest
import numpy as np

from pade.main import *

class CommonTest(unittest.TestCase):

    def test_parse_model_one_var(self):
        m = ModelExpression.parse("treatment")
        self.assertEquals(m.operator, None)
        self.assertEquals(m.variables, ['treatment'])

    def test_parse_model_two_vars(self):
        m = ModelExpression.parse("treatment + sex")
        self.assertEquals(m.operator, "+")
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

    def test_fix_newlines(self):
        self.assertEquals(
            "a\n", fix_newlines("a"))
        self.assertEquals(
            "a b\n", fix_newlines("a\nb"))
        self.assertEquals(
            "a b\nc d\n", fix_newlines("a\nb\n\nc\nd"))
        foo = """\
                   The schema file \"{}\" already exists. If you want to
                   overwrite it, use the --force or -f argument.""".format("foo")
        self.assertEquals("The schema file \"foo\" already exists. If you want to overwrite it, use\nthe --force or -f argument.\n",
                          fix_newlines(foo))

if __name__ == '__main__':
    unittest.main()

