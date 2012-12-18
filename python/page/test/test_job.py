import unittest
import contextlib
import page.main
import os
import numpy as np

from page.test.utils import tempdir

from page.common import Model, ModelExpressionException

class CommonTest(unittest.TestCase):

    def setUp(self):


        self.sample_input_4_class = "sample_data/4_class_testdata_header1.txt"


    def test_table(self):
        schema = page.main.init_schema(self.sample_input_4_class)
        
        with tempdir() as tmp:
            filename = os.path.join(tmp, 'schema.yaml')
            job = page.main.init_job(
                directory=tmp,
                infile=self.sample_input_4_class,
                factors=['treated', 'sex'])
            schema = job.schema

            for c in range(4):
                treated = False if c == 0 or c == 1 else True
                sex = 'male' if c == 0 or c == 2 else 'female'
                for r in range(1, 5):
                    col = 'c{0}r{1}'.format(c, r)
                    schema.set_factor(col, 'treated', treated)
                    schema.set_factor(col, 'sex', sex)

            with open(job.schema_filename, 'w') as out:
                schema.save(out)

            table = job.table

            self.assertEquals(np.shape(table), (16, 1000))

if __name__ == '__main__':
    unittest.main()
