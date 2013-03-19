import unittest
import numpy as np
from pade.test.utils import sample_db
from pade.model import Model, Schema, ModelExpression, ModelExpressionException
from pade.main import init_schema
from StringIO import StringIO
from collections import OrderedDict

class ModelTest(unittest.TestCase):

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

class OtherTest(unittest.TestCase):

    def setUp(self):
        self.sample_input_4_class = "sample_jobs/two_cond_nuisance/sample_data_2_cond_nuisance.txt"

    @property
    def factor_map_treated_sex(self):
        factor_map = { 'treated' : {},
                       'sex'     : {} }
        for c in range(4):
            treated = False if c == 0 or c == 1 else True
            sex = 'male' if c == 0 or c == 2 else 'female'
            for r in range(1, 5):
                col = 'c{0}r{1}'.format(c, r)
                factor_map['treated'][col] = treated
                factor_map['sex'][col] = sex
        return factor_map
        

    def test_table(self):
        schema = init_schema(self.sample_input_4_class)
        
        with sample_db(self.sample_input_4_class,
                        self.factor_map_treated_sex) as job:

            self.assertEquals(np.shape(job.input.table), (1000, 16))


    def test_model_to_layout(self):

        with sample_db(self.sample_input_4_class,
                        self.factor_map_treated_sex) as job:

            # One class
            model = Model(job.schema, 'treated')
            self.assertEquals(
                model.layout,
                [[0, 1, 2, 3, 4, 5, 6, 7],
                 [8, 9, 10, 11, 12, 13, 14, 15]])

            # No classes
            model = Model(job.schema, '')
            self.assertEquals(
                model.layout,
                [[0, 1, 2, 3, 4, 5, 6, 7,
                  8, 9, 10, 11, 12, 13, 14, 15]])

class SchemaTest(unittest.TestCase):

    def setUp(self):
        sample_nums = range(1, 13)

        colnames = ["gene_id"] + ["sample" + str(x) for x in sample_nums]
        
        roles = ['feature_id']
        for i in range(len(sample_nums)):
            roles.append('sample')

        schema = Schema(
            column_names=colnames,
            column_roles=roles)

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
        self.schema = schema

    def test_sample_matches_assignments(self):
        s = self.schema
        self.assertTrue(s.sample_matches_assignments("sample1", { }))
        self.assertTrue(s.sample_matches_assignments("sample1", { 'sex' : 'male' }))
        self.assertFalse(s.sample_matches_assignments("sample1", { 'sex' : 'female' }))

    def test_samples_with_assignments(self):
        self.assertEquals(['sample5', 'sample6'],
                          self.schema.samples_with_assignments({'sex' : 'male',
                                                                'age' : 55}))

    def test_factor_combinations(self):
        expected = [
            (2,  'male',   False), # 0 
            (2,  'male',   True),  # 1 (treated)
            (2,  'female', False), # 1 (sex)
            (2,  'female', True),  # 2 (sex, treated)
            (20, 'male',   False), # 1 (age)
            (20, 'male',   True),  # 2 (age, treated)
            (20, 'female', False), # 2 (age, sex)
            (20, 'female', True),  # 3 (age, sex, treated)
            (55, 'male',   False), # 1 (age)
            (55, 'male',   True),  # 2 (age, treated)
            (55, 'female', False), # 2 (age, sex)
            (55, 'female', True)   # 3 (age, sex, treated)
            ]
        schema = self.schema
        self.assertEquals([('male', False),
                           ('male', True),
                           ('female', False),
                           ('female', True)], 
                          schema.factor_combinations(['sex', 'treated']))
        self.assertEquals([('male',),
                           ('female',)], schema.factor_combinations(['sex']))

        self.assertEquals(expected, schema.factor_combinations())
            

    def test_factors(self):
        schema = self.schema

        self.assertEquals(schema.get_factor("sample1", "sex"), "male")
        self.assertEquals(schema.get_factor("sample1", "age"), 2)
        self.assertEquals(schema.get_factor("sample1", "treated"), True)
        self.assertEquals(schema.get_factor("sample11", "sex"), "female")
        self.assertEquals(schema.get_factor("sample10", "age"), 20)
        self.assertEquals(schema.get_factor("sample10", "treated"), False)

        names = sorted(schema.factors)
        self.assertEquals(names[0], "age")


    def test_yaml(self):
        self.maxDiff = None
        # Save the schema, load it, and save it again. Compare the two
        # versions to make sure they're the same, so that we know we
        # can round-trip.
        out = StringIO()

        self.schema.save(out)
        loaded = Schema.load(out.getvalue())

        out2 = StringIO()
        loaded.save(out2)

        self.assertEquals(out.getvalue(),
                          out2.getvalue())


    def test_baseline_value(self):
        baseline = lambda factor: self.schema.baseline_value(factor)
        self.assertEquals(baseline('sex'), 'male')
        self.assertEquals(baseline('treated'), False)
        self.assertEquals(baseline('age'), 2)

    def test_has_baseline(self):
        baseline = lambda factor: self.schema.baseline_value(factor)
        self.assertTrue(self.schema.has_baseline({'sex' : 'male', 'treated': False}))
        self.assertTrue(self.schema.has_baseline({'sex' : 'male', 'treated': True}))
        self.assertTrue(self.schema.has_baseline({'sex' : 'female', 'treated': False}))
        self.assertFalse(self.schema.has_baseline({'sex' : 'female', 'treated': True}))


    def test_possible_assignments(self):
        self.assertEquals(self.schema.possible_assignments(('sex', 'treated')),
                          [
                OrderedDict([('sex', 'male'), ('treated', False)]),
                OrderedDict([('sex', 'male'), ('treated', True)]),
                OrderedDict([('sex', 'female'), ('treated', False)]),
                OrderedDict([('sex', 'female'), ('treated', True)]),
                ])

    def test_ignore_columns(self):
    
        names = ["gene_id"] 
        roles = ['feature_id']
        for i in range(8):
            names.append('sample_' + str(i))
            if (i % 2) == 0:
                roles.append('sample')
            else:
                roles.append(None)

        schema = Schema(
            column_names=names,
            column_roles=roles)
        self.assertEquals(len(schema.sample_column_names), 4)


        schema.add_factor('treated', [False, True])

        schema.set_factor('sample_0', 'treated', False)
        schema.set_factor('sample_2', 'treated', False)
        schema.set_factor('sample_4', 'treated', True)
        schema.set_factor('sample_6', 'treated', True)

        with self.assertRaises(Exception):
            schema.set_factor('sample_1' + str(i), 'treated', True)

        self.assertEquals(schema.possible_assignments(['treated']),
                          [OrderedDict([('treated', False)]), 
                           OrderedDict([('treated', True )])])

        self.assertEquals(schema.indexes_with_assignments(
                OrderedDict([('treated', False)])),
                          [0, 1])

        self.assertEquals(schema.indexes_with_assignments(
                OrderedDict([('treated', True)])),
                          [2, 3])

        self.assertEquals(schema.samples_with_assignments(
                OrderedDict([('treated', False)])),
                          ['sample_0', 'sample_2'])

        self.assertEquals(schema.samples_with_assignments(
                OrderedDict([('treated', True)])),
                          ['sample_4', 'sample_6'])
        
        out = StringIO()

        schema.save(out)
        loaded = Schema.load(out.getvalue())

        out2 = StringIO()
        loaded.save(out2)

        self.maxDiff = None
        self.assertEquals(out.getvalue(),
                          out2.getvalue())        


if __name__ == '__main__':
    unittest.main()
