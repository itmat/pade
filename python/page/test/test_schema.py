import unittest
import io

from page.main import *

class SchemaTest(unittest.TestCase):

    def setUp(self):
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
        self.schema = schema

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

        names = sorted(schema.factor_names)
        self.assertEquals(names[0], "age")


    def test_yaml(self):
        self.maxDiff = None
        # Save the schema, load it, and save it again. Compare the two
        # versions to make sure they're the same, so that we know we
        # can round-trip.
        out = io.StringIO()

        self.schema.save(out)
        loaded = Schema.load(out.getvalue())

        out2 = io.StringIO()
        loaded.save(out2)

        self.assertEquals(out.getvalue(),
                          out2.getvalue())

    def test_sample_groups(self):
        groups = self.schema.sample_groups(["sex"])
        self.assertEquals(groups,
                          OrderedDict({ ("female",) : range(6, 12),
                                        ("male",)   : range(0, 6) }))

        groups = self.schema.sample_groups(["age"])
        self.assertEquals(groups,
                          { (2,)  : [0, 1, 6, 7],
                            (20,) : [2, 3, 8, 9],
                            (55,) : [4, 5, 10, 11] })

        groups = self.schema.sample_groups(["treated"])
        self.assertEquals(groups,
                          { (False,) : [1, 3, 5, 7, 9, 11],
                            (True,)  : [0, 2, 4, 6, 8, 10] })

        groups = self.schema.sample_groups(["age", "sex", "treated"])
        self.assertEquals(
            groups,
            OrderedDict( 
                [
                    ((2,  "male",    False) ,  [  1 ]),
                    ((2,  "male",    True)  ,  [  0 ]),
                    ((2,  "female",  False) ,  [  7 ]),
                    ((2,  "female",  True)  ,  [  6 ]),
                    ((20, "male",    False) ,  [  3 ]),
                    ((20, "male",    True)  ,  [  2 ]),
                    ((20, "female",  False) ,  [  9 ]),
                    ((20, "female",  True)  ,  [  8 ]),
                    ((55, "male",    False) ,  [  5 ]),
                    ((55, "male",    True)  ,  [  4 ]),
                    ((55, "female",  False) ,  [ 11 ]),
                    ((55, "female",  True)  ,  [ 10 ]),
]))

    def test_factor_value_shape(self):
        model = lambda(expr): Model(self.schema, expr)

        shape = lambda(expr): model(expr).factor_value_shape()

        self.assertEquals(shape('sex'), (2,))
        self.assertEquals(shape('treated'), (2,))
        self.assertEquals(shape('age'), (3,))
        self.assertEquals(shape('sex * age'), (2, 3))

    def test_col_names(self):
        model = lambda(expr): Model(self.schema, expr)
        self.assertEquals(
            model_col_names(model('sex')),
            ['intercept', 'sex=female'])

        self.assertEquals(
            model_col_names(model('sex * age')),
            ['intercept',
             'age=20',
             'age=55',
             'sex=female',
             'sex=female, age=20',
             'sex=female, age=55'])

    def test_dummy_vars(self):
        np.testing.assert_equal(
            self.schema.dummy_vars({"sex" : "male"}),
            np.array([0], bool))
        np.testing.assert_equal(
            self.schema.dummy_vars({"sex" : "female"}),
            np.array([1], bool))
        np.testing.assert_equal(
            self.schema.dummy_vars({"treated" : False}),
            np.array([0], bool))
        np.testing.assert_equal(
            self.schema.dummy_vars({"treated" : True}),
            np.array([1], bool))
        
    def test_model_dummy_vars(self):
        model = Model(self.schema, 'age + treated')

        expected_vars = np.array([
                [ 1, 0, 0, 1],
                [ 1, 0, 0, 0],
                [ 1, 1, 0, 1],
                [ 1, 1, 0, 0],
                [ 1, 0, 1, 1],
                [ 1, 0, 1, 0],
                [ 1, 0, 0, 1],
                [ 1, 0, 0, 0],
                [ 1, 1, 0, 1],
                [ 1, 1, 0, 0],
                [ 1, 0, 1, 1],
                [ 1, 0, 1, 0],
                ], bool)

        (vars, indexes) = self.schema.dummy_vars_and_indexes(
            ['age', 'treated'])

        np.testing.assert_equal(vars, expected_vars)
        print "Indexes is ", indexes
        np.testing.assert_equal(indexes,
                                np.arange(12, dtype=int))

    def test_baseline_value(self):
        baseline = lambda factor: self.schema.baseline_value(factor)
        self.assertEquals(baseline('sex'), 'male')
        self.assertEquals(baseline('treated'), False)
        self.assertEquals(baseline('age'), 2)

#     def test_model_dummy_vars_1(self):
#         model = Model(self.schema, 'age + treated')

#         expected_vars = np.array([
#                 [ 1, 0, 0, 1, 0, 0],
#                 [ 1, 0, 0, 0, 0, 0],
#                 [ 1, 1, 0, 1, 1, 0],
#                 [ 1, 1, 0, 0, 0, 0],
#                 [ 1, 0, 1, 1, 0, 1],
#                 [ 1, 0, 1, 0, 0, 0],
#                 [ 1, 0, 0, 1, 0, 0],
#                 [ 1, 0, 0, 0, 0, 0],
#                 [ 1, 1, 0, 1, 1, 0],
#                 [ 1, 1, 0, 0, 0, 0],
#                 [ 1, 0, 1, 1, 0, 1],
#                 [ 1, 0, 1, 0, 0, 0],
#                 ], bool)

#         (vars, indexes) = self.schema.dummy_vars_and_indexes(
#             ['age', 'treated'],
#             interactions=1)

#         np.testing.assert_equal(vars, expected_vars)
#         print "Indexes is ", indexes
#         np.testing.assert_equal(indexes,
#                                 np.arange(12, dtype=int))

if __name__ == '__main__':
    unittest.main()
    
