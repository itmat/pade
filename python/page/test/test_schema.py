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



unittest.main()
