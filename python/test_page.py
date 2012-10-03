import page

import unittest
from numpy import *

class PageTest(unittest.TestCase):

    def setUp(self):
        self.v1 = array([float(x) for x in [1, 6, 5, 3, 8, 9, 6, 3, 6, 8]])
        self.v2 = array([float(x) for x in [7, 4, 9, 6, 2, 4, 7, 4, 2, 1]])

    def test_compute_s(self):
        s = page.compute_s(self.v1, self.v2, 10, 10)
        self.assertAlmostEqual(s, 2.57012753682683)
        
unittest.main()
