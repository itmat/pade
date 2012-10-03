import page

import unittest
from numpy import *
import numpy.ma as ma

class PageTest(unittest.TestCase):

    def setUp(self):
        self.v1 = ma.masked_array([float(x) for x in [1, 6, 5, 3, 8, 9, 6, 3, 6, 8]], ma.nomask)
        self.v2 = ma.masked_array([float(x) for x in [7, 4, 9, 6, 2, 4, 7, 4, 2, 1]], ma.nomask)
        self.config = page.Config({})
        self.config.infile = 'sample_data/4_class_testdata_header1.txt'

    def test_compute_s(self):
        s = page.compute_s(self.v1, self.v2, 10, 10)
        self.assertAlmostEqual(s, 2.57012753682683)

    def test_default_alpha(self):
        data = page.load_input(self.config)
        alphas = page.find_default_alpha(data)

        self.assertAlmostEqual(alphas[0], 1.62026604316528)
        self.assertAlmostEqual(alphas[1], 1.61770701155527)
        self.assertAlmostEqual(alphas[2], 1.60540468969643)

        page.compute_s(data.table[:,(0,1,2,3)],
                         data.table[:,(4,5,6,7)], 10, 10)
        
unittest.main()

2.71876197315371

#Mean is 2.29140221289693
#SD is 1.66873784599192
#Mean is 2.28778319568751
#SD is 1.67061575354633
#Mean is 2.27038508526607
#SD is 1.66848321913367
#701
#693
#693

# alpha is  
