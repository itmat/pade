import unittest
import numpy as np

from pade.main import *

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
        
        means = get_group_means(self.schema, data, self.schema.factors)
        np.testing.assert_almost_equal(means, expected_means)

    def test_coeffs_with_interaction(self):

        model = Model(self.schema, "treated * sex")

        data = np.array(
            [[ 1, 1, 3, 3, 4, 4,  6,  6 ],
             [ 1, 1, 3, 3, 4, 4, 10, 10 ]])

        expected_coeffs = np.array([
            [ 1, 3, 2, 0],
            [ 1, 3, 2, 4]])

        expected_labels = ({}, {'treated' : True}, {'sex' : 'female'}, { 'sex': 'female', 'treated' : True})

        fitted = model.fit(data)

        self.assertEquals(expected_labels, fitted.labels)
        np.testing.assert_almost_equal(expected_coeffs, fitted.params)

    def test_coeffs_no_interaction(self):
        
        model = Model(self.schema, "treated + sex")

        data = np.array([
                [ 1, 1, 3, 3, 4, 4,  6,  6 ],
                [ 1, 1, 3, 3, 4, 4, 10, 10 ]])
                                       
        expected_coeffs = np.array([
                [ 1.0,  3, 2],
                [ 0.0,  5, 4]])
            

        expected_labels = ({}, {'treated' : True}, {'sex' : 'female'})
        fitted = model.fit(data)
        self.assertEquals(expected_labels, fitted.labels)
        np.testing.assert_almost_equal(expected_coeffs, fitted.params)

    def test_model_dummy_vars_1(self):

        sample_nums = range(1, 13)

        colnames = ["gene_id"] + ["sample" + str(x) for x in sample_nums]
        
        is_feature_id = [True]  + [False for x in sample_nums]
        is_sample     = [False] + [True  for x in sample_nums]

        schema = Schema(
            column_names=colnames,
            is_feature_id=is_feature_id,
            is_sample=is_sample)

        schema.add_factor('age', [2, 20, 55])
        schema.add_factor('sex', ['male', 'female'])
        schema.add_factor('treated', [False, True])

        counter = 0

        self.assertEquals(schema.sample_num("sample1"), 0);
        self.assertEquals(schema.sample_num("sample7"), 6);

        for sex in ['male', 'female']:
            for age in [2, 20, 55]:
                for treated in [True, False]:
                    counter += 1
                    name = "sample" + str(counter)
                    schema.set_factor(name, 'sex',     sex)
                    schema.set_factor(name, 'age',     age)
                    schema.set_factor(name, 'treated', treated)

        dummies = dummy_vars(schema, ['age', 'treated'], level=2)

        expected = DummyVarTable(
            ({}, {'age': 20}, {'age': 55}, {'treated': True}, {'age': 20, 'treated': True}, {'age': 55, 'treated': True}),
            [
                DummyVarAssignment(factor_values=(2, False),  bits=(True, False, False, False, False, False), indexes=['sample2', 'sample8']),
                DummyVarAssignment(factor_values=(2, True),   bits=(True, False, False, True, False, False), indexes=['sample1', 'sample7']),
                DummyVarAssignment(factor_values=(20, False), bits=(True, True, False, False, False, False), indexes=['sample4', 'sample10']),
                DummyVarAssignment(factor_values=(20, True),  bits=(True, True, False, True, True, False), indexes=['sample3', 'sample9']),
                DummyVarAssignment(factor_values=(55, False), bits=(True, False, True, False, False, False), indexes=['sample6', 'sample12']),
                DummyVarAssignment(factor_values=(55, True), bits=(True, False, True, True, False, True), indexes=['sample5', 'sample11'])])

        self.assertEquals(dummies, expected)

