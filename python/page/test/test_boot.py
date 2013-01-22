import numpy as np
import unittest
from page.main import *

class BootTest(unittest.TestCase):

    def test_sample_indexes_no_grouping(self):
        
        layout = [np.arange(6)]
        
        idxs = sample_indexes(layout, 1000)

        self.assertEquals(np.shape(idxs),
                          (1000, 6))

        self.assertTrue(np.all(idxs < 6) and np.all(idxs >= 0))



    def test_sample_indexes_with_grouping(self):
        
        layout = [[ 0, 1, 2 ],
                  [ 3, 4, 5 ]]
        
        idxs = sample_indexes(layout, 10)
        print idxs
        self.assertEquals(np.shape(idxs),
                          (10, 6))

        group1 = idxs[:, :3]
        group2 = idxs[:,  3:]

        self.assertTrue(np.min(group1) >= 0)
        self.assertTrue(np.max(group1) < 3)
        self.assertTrue(np.min(group2) >= 3)
        self.assertTrue(np.max(group2) < 6)


