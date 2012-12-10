import numpy as np
import unittest
from page.stats import Tstat, Ftest
import page.core as page

class StatTest(unittest.TestCase):

    def setUp(self):
        self.ftest_in = np.array([
                [ 6.0,  8.0, 13.0],
                [ 8.0, 12.0,  9.0],
                [ 4.0,  9.0, 11.0],
                [ 5.0, 11.0,  8.0],
                [ 3.0,  6.0,  7.0],
                [ 4.0,  8.0, 12.0]])


    def test_tstat(self):
        data = np.array(
            [[[2.410962, 1.897421, 2.421239, 1.798668],
              [2.410962, 1.897421, 2.421239, 1.798668]],
             [[0.90775,  0.964438, 1.07578,  1.065872],
              [0.90775,  0.964438, 1.07578,  1.065872]]])

        alpha = 1.62026604316528 * Tstat.TUNING_PARAM_RANGE_VALUES[4]
        stat = Tstat(alpha)
        result = stat.compute(data)

        expected = [1.51898640652018,
                    1.51898640652018]

        self.assertAlmostEqual(sum(result),
                               sum(expected),
                               )

    def test_ftest(self):

        self.assertAlmostEqual(9.26470588235, Ftest().compute(self.ftest_in))

        a2 = np.concatenate((self.ftest_in,
                             self.ftest_in)).reshape((2, 6, 3))
        print Ftest().compute(a2)


if __name__ == '__main__':
    unittest.main()
