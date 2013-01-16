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

        schema.add_factor('age', object)
        schema.add_factor('sex', object)
        schema.add_factor('treated', object)

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
                          { ("sex", "female") : range(6, 12),
                            ("sex", "male")   : range(0, 6) })

        groups = self.schema.sample_groups(["age"])
        self.assertEquals(groups,
                          { ("age", 2)  : [0, 1, 6, 7],
                            ("age", 20) : [2, 3, 8, 9],
                            ("age", 55) : [4, 5, 10, 11] })

        groups = self.schema.sample_groups(["treated"])
        self.assertEquals(groups,
                          { ("treated", False) : [1, 3, 5, 7, 9, 11],
                            ("treated", True)  : [0, 2, 4, 6, 8, 10] })

        groups = self.schema.sample_groups(["sex", "age", "treated"])
        self.assertEquals(groups,
                          { 
                ("sex", "male", 
                 "age", 2, 
                 "treated", True) : [ 0 ],
                ("sex", "male",
                 "age", 2,
                 "treated", False) : [ 1 ],

                ("sex", "male", 
                 "age", 20, 
                 "treated", True) : [ 2 ],
                ("sex", "male",
                 "age", 20,
                 "treated", False) : [ 3 ],

                ("sex", "male", 
                 "age", 55, 
                 "treated", True) : [ 4 ],
                ("sex", "male",
                 "age", 55,
                 "treated", False) : [ 5 ],

                ("sex", "female", 
                 "age", 2, 
                 "treated", True) : [ 6 ],
                ("sex", "female",
                 "age", 2,
                 "treated", False) : [ 7 ],

                ("sex", "female", 
                 "age", 20, 
                 "treated", True) : [ 8 ],
                ("sex", "female",
                 "age", 20,
                 "treated", False) : [ 9 ],

                ("sex", "female", 
                 "age", 55, 
                 "treated", True) : [ 10 ],
                ("sex", "female",
                 "age", 55,
                 "treated", False) : [ 11 ]

                })
                

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


if __name__ == '__main__':
    unittest.main()
    
