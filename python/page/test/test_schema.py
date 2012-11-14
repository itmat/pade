import unittest
import io

from page.schema import Schema

class SchemaTest(unittest.TestCase):

    def setUp(self):
        sample_nums = range(1, 13)

        colnames = ["gene_id"] + ["sample" + str(x) for x in sample_nums]
        
        is_feature_id = [True]  + [False for x in sample_nums]
        is_sample     = [False] + [True  for x in sample_nums]

        schema = Schema(
            attributes=[
                ('name',    object),
                ('sex',     object),
                ('age',     'int'),
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
        self.schema = schema

    def test_attributes(self):
        schema = self.schema

        self.assertEquals(schema.get_attribute("sample1", "sex"), "male")
        self.assertEquals(schema.get_attribute("sample1", "age"), 2)
        self.assertEquals(schema.get_attribute("sample1", "treated"), True)
        self.assertEquals(schema.get_attribute("sample11", "sex"), "female")
        self.assertEquals(schema.get_attribute("sample10", "age"), 20)
        self.assertEquals(schema.get_attribute("sample10", "treated"), False)

        names = sorted(schema.attribute_names)
        self.assertEquals(names[0], "age")


    def test_yaml(self):
        self.maxDiff = None
        # Save the schema, load it, and save it again. Compare the two
        # versions to make sure they're the same, so that we know we
        # can round-trip.
        out = io.StringIO()
        self.schema.save(out)

        with open("sample_data/test_infile.tab") as infile:
            loaded = Schema.load(out.getvalue(), infile)

        out2 = io.StringIO()
        loaded.save(out2)

        self.assertEquals(out.getvalue(),
                          out2.getvalue())

    def test_sample_groups(self):
        groups = self.schema.sample_groups("sex")
        self.assertEquals(groups,
                          { "female" : range(6, 12),
                            "male"   : range(0, 6) })

        groups = self.schema.sample_groups("age")
        self.assertEquals(groups,
                          { 2  : [0, 1, 6, 7],
                            20 : [2, 3, 8, 9],
                            55 : [4, 5, 10, 11] })

        groups = self.schema.sample_groups("treated")
        self.assertEquals(groups,
                          { False : [1, 3, 5, 7, 9, 11],
                            True  : [0, 2, 4, 6, 8, 10] })

