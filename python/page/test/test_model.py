import unittest
import numpy as np

from page.main import *

class ModelTest(unittest.TestCase):

    def setUp(self):
        sample_nums = range(1, 9)

        colnames = ["gene_id"] + ["sample" + str(x) for x in sample_nums]
        
        is_feature_id = [True]  + [False for x in sample_nums]
        is_sample     = [False] + [True  for x in sample_nums]

        schema = Schema(
            column_names=colnames,
            is_feature_id=is_feature_id,
            is_sample=is_sample)

        schema.add_factor('treated', [False, True])
        schema.add_factor('sex', ['male', 'female'])
        
        factor_table = [
                ('sample1', 'male',   False),
                ('sample2', 'male',   False),
                ('sample3', 'female', False),
                ('sample4', 'female', False),
                ('sample5', 'male',   True),
                ('sample6', 'male',   True),
                ('sample7', 'female', True),
                ('sample8', 'female', True)]

        for row in factor_table:
            (name, sex, treated) = row
            schema.set_factor(name, 'sex', sex)
            schema.set_factor(name, 'treated', treated)

        self.schema = schema

    def test_group_means(self):

        data = np.array(
            [[ 1, 1, 3, 3, 4, 4,  6,  6 ],
             [ 1, 1, 3, 3, 4, 4, 10, 10 ]])

        expected_means = np.array([
            [ 1, 3, 4,  6],
            [ 1, 3, 4, 10]])
        
        means = get_group_means(self.schema, data)
        np.testing.assert_almost_equal(means, expected_means)

    def test_coeffs_with_interaction(self):

        model = Model(self.schema, "treated * sex")

        data = np.array(
            [[ 1, 1, 3, 3, 4, 4,  6,  6 ],
             [ 1, 1, 3, 3, 4, 4, 10, 10 ]])

        expected_coeffs = np.array([
            [ 1, 2, 3, 0],
            [ 1, 2, 3, 4]])

        coeffs = find_coefficients_with_interaction(model, data)

        np.testing.assert_almost_equal(expected_coeffs, coeffs)

    def test_coeffs_no_interaction(self):
        
        model = Model(self.schema, "sex + treated")

        data = np.array([
                [ 1, 1, 3, 3, 4, 4,  6,  6 ],
                [ 1, 1, 3, 3, 4, 4, 10, 10 ]])
                                       
        expected_coeffs = np.array([
                [ 1.0,  2.0, 3.0],
                [ 0.0,  4.0, 5.0 ]])
            
        coeffs = find_coefficients_no_interaction(model, data)

        np.testing.assert_almost_equal(expected_coeffs, coeffs)


#    def  test_factor_vars(self):
#         vars = [True, True, True]

#         self.assertEquals([(True, False)],
#                           factor_vars((True, True)))

#         self.assertEquals([(True, False, False)],
#                           factor_vars((True, False, True)))

#         self.assertEquals([(True, False, False),
#                            (True, False, True),
#                            (True, True,  False)],
#                            factor_vars((True, True, True)))

#         self.assertEquals([(True, False, False, False),
#                            (True, False, False, True),
#                            (True, True,  False, False)],
#                           factor_vars((True, True, False, True)))

#         self.assertEquals([(True, False, False, False),
#                            (True, False, False, True),
#                            (True, False, True,  False),
#                            (True, False, True,  True),
#                            (True, True,  False, False),
#                            (True, True,  False, True),
#                            (True, True,  True, False)],
#                           factor_vars((True, True, True, True)))
                         
                           
