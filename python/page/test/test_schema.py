import sys, os

sys.path.insert(0, os.path.dirname(__file__) + "/../..")
import unittest
import page
from page import Schema, SchemaException

class SchemaTest(unittest.TestCase):

    def test_newschema(self):

        sample_nums = range(1, 13)

        colnames = ["gene_id"] + ["sample" + str(x) for x in sample_nums]
        
        is_feature_id = [True]  + [False for x in sample_nums]
        is_sample     = [False] + [True  for x in sample_nums]

        schema = page.NewSchema(
            attributes=[
                ('name', 'S100'),
                ('sex', 'S100'),
                ('age', 'int'),
                ('treated', 'bool')],
            column_names=colnames,
            is_feature_id=is_feature_id,
            is_sample=is_sample)

        counter = 0

        self.assertEquals(schema.sample_num("sample1"), 0);
        self.assertEquals(schema.sample_num("sample7"), 6);

        for sex in ['male', 'female']:
            for age in [2, 20, 55]:
                for treated in [True, False]:
                    counter += 1
                    name = "sample" + str(counter)
                    schema.set_attribute(name, 'sex',     sex)
                    schema.set_attribute(name, 'age',     age)
                    schema.set_attribute(name, 'treated', treated)

        self.assertEquals(schema.get_attribute("sample1", "sex"), "male")
        self.assertEquals(schema.get_attribute("sample1", "age"), 2)
        self.assertEquals(schema.get_attribute("sample1", "treated"), True)
        self.assertEquals(schema.get_attribute("sample11", "sex"), "female")
        self.assertEquals(schema.get_attribute("sample10", "age"), 20)
        self.assertEquals(schema.get_attribute("sample10", "treated"), False)

unittest.main()
