import unittest
import contextlib
import page
import os
import numpy as np

from tests.utils import sample_job
from page import *

class CommonTest(unittest.TestCase):

    def setUp(self):


        self.sample_input_4_class = "sample_data/4_class_testdata_header1.txt"

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
        schema = page.init_schema(self.sample_input_4_class)
        
        with sample_job(self.sample_input_4_class,
                        self.factor_map_treated_sex) as job:

            self.assertEquals(np.shape(job.table), (16, 1000))
            print job.schema.sample_groups(['treated'])
            print job.schema.sample_groups(['sex'])


    def test_model_to_layout(self):
        model = ModelExpression(variables=['treated'])
        with sample_job(self.sample_input_4_class,
                        self.factor_map_treated_sex) as job:

            # One class
            layout_map = page.model_to_layout_map(job.schema, model)
            self.assertEquals(
                layout_map.keys(),
                [('treated', True),
                 ('treated', False)])
            self.assertEquals(
                layout_map.values(),
                [[8, 9, 10, 11, 12, 13, 14, 15],
                 [0, 1, 2, 3, 4, 5, 6, 7]])

            # No classes
            model = ModelExpression()
            layout_map = page.model_to_layout_map(job.schema, model)
            self.assertEquals(
                layout_map.keys(),
                [tuple()])
            self.assertEquals(
                layout_map.values(),
                [[0, 1, 2, 3, 4, 5, 6, 7,
                  8, 9, 10, 11, 12, 13, 14, 15]])


if __name__ == '__main__':
    unittest.main()
    
