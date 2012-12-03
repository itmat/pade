import numpy as np
import unittest
from page.stats import Tstat
import page.core as page

class StatTest(unittest.TestCase):
    def test_a_tstat(self):
        v1 = [[2.410962, 1.897421, 2.421239, 1.798668],
              [2.410962, 1.897421, 2.421239, 1.798668]]
        v2 = [[0.90775,  0.964438, 1.07578,  1.065872],
              [0.90775,  0.964438, 1.07578,  1.065872]]

        v1 = np.array(v1)
        v2 = np.array(v2)
        alpha = 1.62026604316528 * Tstat.TUNING_PARAM_RANGE_VALUES[4]
        stat = Tstat(alpha)
        result = stat.compute((v1, v2))
        print result
        expected = [1.51898640652018,
                    1.51898640652018]

        self.assertAlmostEqual(sum(result),
                               sum(expected),
                               )

