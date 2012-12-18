import unittest
import numpy as np

from page.common import Model, ModelExpressionException, add_condition_axis

class CommonTest(unittest.TestCase):

    def test_parse_model_one_var(self):
        m = Model.parse("treatment")
        self.assertEquals(m.prob, None)
        self.assertEquals(m.variables, ['treatment'])

    def test_parse_model_two_vars(self):
        m = Model.parse("treatment + sex")
        self.assertEquals(m.prob, Model.PROB_MARGINAL)
        self.assertEquals(m.variables, ['treatment', 'sex'])
        
    def test_parse_op(self):
        with self.assertRaises(ModelExpressionException):
            Model.parse("+")

    def test_parse_var_var(self):
        with self.assertRaises(ModelExpressionException):
            Model.parse("treatment sex")

    def test_parse_var_op_var_op_var(self):
        with self.assertRaises(ModelExpressionException):
            Model.parse("treatment + sex + batch")

    def test_str(self):
        self.assertEquals(
            str(Model.parse("treatment")), "treatment")
        self.assertEquals(
            str(Model.parse("treatment * sex")), "treatment * sex")

    def test_add_condition_axis(self):
        data = add_condition_axis(np.arange(6, dtype=int),
                                  ([0,1,2],[3,4,5]))
        expected = np.array([[0,1,2],[3,4,5]])
        self.assertTrue(
            np.all(expected - data == 0))
        


if __name__ == '__main__':
    unittest.main()

