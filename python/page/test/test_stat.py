import numpy as np
import unittest
from page.stats import Ttest, Ftest
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
            [
                [
                    [2.410962, 0.90775],
                    [1.897421, 0.964438],
                    [2.421239, 1.07578],
                    [1.798668, 1.065872]],
                [
                    [2.410962, 0.90775],
                    [1.897421, 0.964438],
                    [2.421239, 1.07578],
                    [1.798668, 1.065872]],
                ])
        data = np.swapaxes(data, 0, 2)
        alpha = 1.62026604316528 * Ttest.TUNING_PARAM_RANGE_VALUES[4]
        stat = Ttest(alpha)
        result = stat.compute(data)

        expected = [1.51898640652018,
                    1.51898640652018]

        self.assertAlmostEqual(sum(result),
                               sum(expected),
                               )

    def test_3d_tstat(self):
        data = np.array(
            [
                [
                    [2.410962, 0.90775],
                    [1.897421, 0.964438],
                    [2.421239, 1.07578],
                    [1.798668, 1.065872]],
                [
                    [2.410962, 0.90775],
                    [1.897421, 0.964438],
                    [2.421239, 1.07578],
                    [1.798668, 1.065872]],
                [
                    [2.410962, 0.90775],
                    [1.897421, 0.964438],
                    [2.421239, 1.07578],
                    [1.798668, 1.065872]]])
        data = np.swapaxes(data, 0, 2)
        alpha = 1.62026604316528 * Ttest.TUNING_PARAM_RANGE_VALUES[4]
        stat = Ttest(alpha)
        result = stat.compute(data)

        expected = [1.51898640652018,
                    1.51898640652018,
                    1.51898640652018,
                    ]

        self.assertAlmostEqual(sum(result),
                               sum(expected),
                               )

    def test_2d_tstat(self):
        data = np.array(
            [
                [2.410962, 0.90775],
                [1.897421, 0.964438],
                [2.421239, 1.07578],
                [1.798668, 1.065872]])
        data = np.swapaxes(data, 0, 1)
        alpha = 1.62026604316528 * Ttest.TUNING_PARAM_RANGE_VALUES[4]
        stat = Ttest(alpha)
        result = stat.compute(data)

        expected = 1.51898640652018
        self.assertAlmostEqual(result, expected)


    def test_ftest(self):

        expected = 9.26470588235
        self.assertAlmostEqual(expected, Ftest().compute(self.ftest_in))

        a2 = np.concatenate((self.ftest_in,
                             self.ftest_in)).reshape((2, 6, 3))
        a2 = np.swapaxes(a2, 0, 1)
        a2 = np.swapaxes(a2, 1, 2)
        got = Ftest().compute(a2)
        self.assertEqual(np.shape(got), (2,))
        self.assertAlmostEqual(got[0], expected)
        self.assertAlmostEqual(got[1], expected)

if __name__ == '__main__':
    unittest.main()
