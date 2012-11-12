import sys, os

sys.path.insert(0, os.path.dirname(__file__) + "/../..")
import unittest
import page
from page import Schema, SchemaException

class SchemaTest(unittest.TestCase):

    def test_add_sample(self):
        schema = Schema()
        schema.add_sample('sample1', 1)
        schema.add_sample('sample2', 2)
        self.assertTrue('sample1' in schema.column_index)
        self.assertTrue('sample2' in schema.column_index)

    def test_add_factor(self):
        schema = page.Schema()
        schema.add_factor('sex', values=['male', 'female'])
        self.assertEquals(schema.factor_values['sex'][0], 'male')

    def test_set_column_factor_value(self):
        schema = page.Schema()
        schema.add_sample('sample1', 1)
        schema.add_sample('sample2', 2)

        schema.add_factor('sex', values=['male', 'female'])
        schema.set_column_factor('sample1', 'sex', 'male')
        schema.set_column_factor('sample2', 'sex', 'female')

        self.assertEquals(schema.get_column_factor('sample1', 'sex'), 'male')
        self.assertEquals(schema.get_column_factor('sample2', 'sex'), 'female')

        self.assertRaises(SchemaException,
                          schema.set_column_factor, 'foobar', 'sex', 'male')
        self.assertRaises(SchemaException,
                          schema.set_column_factor, 'sample1', 'foobar', 'male')
        self.assertRaises(SchemaException,
                          schema.set_column_factor, 'sample1', 'sex', 'foobar')


    def test_newschema(self):

        sample_nums = range(1, 13)

        colnames = ["gene_id"] + ["sample" + str(x) for x in sample_nums]
        
        is_feature_id = [True]  + [False for x in sample_nums]
        is_sample     = [False] + [True  for x in sample_nums]

        schema = page.NewSchema(
            attributes=[
                'name', 'sex', 'age', 'treated'],
            column_names=colnames,
            is_feature_id=is_feature_id,
            is_sample=is_sample)

        counter = 0

        self.assertEquals(schema.sample_num("sample1"), 0);
        self.assertEquals(schema.sample_num("sample7"), 6);

        for sex in ['male', 'female']:
            for age in ['child', 'adult', 'senior']:
                for treated in ['yes', 'no']:
                    counter += 1
                    name = "sample" + str(counter)
                    schema.set_attribute(name, 'sex',     sex)
                    schema.set_attribute(name, 'age',     age)
                    schema.set_attribute(name, 'treated', treated)

        self.assertEquals(schema.get_attribute("sample1", "sex"), "male")
        self.assertEquals(schema.get_attribute("sample1", "age"), "child")
        self.assertEquals(schema.get_attribute("sample1", "treated"), "yes")
        self.assertEquals(schema.get_attribute("sample11", "sex"), "female")
        self.assertEquals(schema.get_attribute("sample10", "age"), "adult")
        self.assertEquals(schema.get_attribute("sample10", "treated"), "no")

unittest.main()
