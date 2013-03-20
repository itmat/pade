import numpy as np
import unittest
from pade.stat import *
from pade.layout import random_orderings, num_orderings, all_orderings

def pairedOrderings(n, R):
    idxs = np.arange(2 * n)

    cond_layout = [ range(0, 2 * n, 2),
                    range(1, 2 * n, 2) ]
    
    reduced_layout = idxs.reshape((n, 2))
    return list(random_orderings(cond_layout, reduced_layout, 100))

class StatTest(unittest.TestCase):

    def setUp(self):
        self.ftest_in = np.array([
                 6.0,  8.0, 13.0,
                 8.0, 12.0,  9.0,
                 4.0,  9.0, 11.0,
                 5.0, 11.0,  8.0,
                 3.0,  6.0,  7.0,
                 4.0,  8.0, 12.0])

        self.infile = 'sample_data/4_class_testdata_header1.txt'

    def test_ftest_no_tuning_params(self):

        expected = 9.26470588235
        ftest = FStat(
            condition_layout=[range(i, 18, 3) for i in range(3)],
            block_layout=[range(18)])
        self.assertAlmostEqual(expected, ftest(self.ftest_in))

        a2 = np.concatenate((self.ftest_in,
                             self.ftest_in)).reshape((2, 18))
        print a2
        got = ftest(a2)
        self.assertEqual(np.shape(got), (2,))
        self.assertAlmostEqual(got[0], expected)
        self.assertAlmostEqual(got[1], expected)

    def test_ftest_with_tuning_params(self):

        alphas = np.array([0.0, 0.01, 0.1, 1, 3])

        expected = 9.26470588235
        ftest = FStat(
            condition_layout=[range(i, 18, 3) for i in range(3)],
            block_layout=[range(18)],
            alphas=alphas)

        self.assertAlmostEqual(expected, ftest(self.ftest_in)[0])

    def test_num_orderings(self):

        def assertOrderings(full, reduced, expected):
            got = num_orderings(full, reduced)
            self.assertEquals(got, expected)

        assertOrderings([[0]], None, 1)
        assertOrderings([[0, 1]],   None, 1)
        assertOrderings([[0], [1]], None, 2)
        assertOrderings([[0, 1], [2, 3]], None, 6)

        assertOrderings([[0, 1], [2, 3], [4, 5]], None, 90)

        assertOrderings([[0, 1], [2, 3], [4, 5], [6, 7]],
                           [[0, 1,   2, 3], [4, 5,   6, 7]],
                           36)


    def test_all_orderings(self):
        def assertOrderings(full, reduced, expected):
            got = all_orderings(full, reduced)
            self.assertEquals(list(got), expected)        
        
        assertOrderings([[0]], 
                        [[0]],
                        [[0]])
        
        assertOrderings([[0], [1]], 
                        [[0, 1]],
                        [[0, 1], [1, 0]])
        
        assertOrderings([[0, 1], [2, 3]],
                        [[0, 1,   2, 3]],
                        [[0, 1,   2, 3],
                         [0, 2,   1, 3],
                         [0, 3,   1, 2],
                         [1, 2,   0, 3],
                         [1, 3,   0, 2],
                         [2, 3,   0, 1]])
        
        assertOrderings(
            [[0, 1, 4, 5], [2, 3, 6, 7]],
            [[0, 1,   2, 3], [4, 5,   6, 7]],
            [
                [0, 1,   2, 3,   4, 5,   6, 7],
                [0, 1,   2, 3,   4, 6,   5, 7],
                [0, 1,   2, 3,   4, 7,   5, 6],
                 [0, 1,   2, 3,   5, 6,   4, 7],
                 [0, 1,   2, 3,   5, 7,   4, 6],
                 [0, 1,   2, 3,   6, 7,   4, 5],

                 [0, 2,   1, 3,   4, 5,   6, 7],
                 [0, 2,   1, 3,   4, 6,   5, 7],
                 [0, 2,   1, 3,   4, 7,   5, 6],
                 [0, 2,   1, 3,   5, 6,   4, 7],
                 [0, 2,   1, 3,   5, 7,   4, 6],
                 [0, 2,   1, 3,   6, 7,   4, 5],

                 [0, 3,   1, 2,   4, 5,   6, 7],
                 [0, 3,   1, 2,   4, 6,   5, 7],
                 [0, 3,   1, 2,   4, 7,   5, 6],
                 [0, 3,   1, 2,   5, 6,   4, 7],
                 [0, 3,   1, 2,   5, 7,   4, 6],
                 [0, 3,   1, 2,   6, 7,   4, 5],
                     
                 [1, 2,   0, 3,   4, 5,   6, 7],
                 [1, 2,   0, 3,   4, 6,   5, 7],
                 [1, 2,   0, 3,   4, 7,   5, 6],
                 [1, 2,   0, 3,   5, 6,   4, 7],
                 [1, 2,   0, 3,   5, 7,   4, 6],
                 [1, 2,   0, 3,   6, 7,   4, 5],

                 [1, 3,   0, 2,   4, 5,   6, 7],
                 [1, 3,   0, 2,   4, 6,   5, 7],
                 [1, 3,   0, 2,   4, 7,   5, 6],
                 [1, 3,   0, 2,   5, 6,   4, 7],
                 [1, 3,   0, 2,   5, 7,   4, 6],
                 [1, 3,   0, 2,   6, 7,   4, 5],

                 [2, 3,   0, 1,   4, 5,   6, 7],
                 [2, 3,   0, 1,   4, 6,   5, 7],
                 [2, 3,   0, 1,   4, 7,   5, 6],
                 [2, 3,   0, 1,   5, 6,   4, 7],
                 [2, 3,   0, 1,   5, 7,   4, 6],
                 [2, 3,   0, 1,   6, 7,   4, 5]])
        
    def test_random_orderings(self):
        arrs = random_orderings([[0]], [[0]], 10)
        self.assertEquals(list(arrs), [ [ 0 ] ])

        arrs = random_orderings([[0], [1]], [[0, 1]], 10)
        self.assertEquals(list(arrs), [[0, 1], [1, 0]])

        arrs = random_orderings([[0, 1], [2, 3]], [[0, 1, 2, 3]], 10)
        self.assertEquals(list(arrs),
                           [[0, 1, 2, 3], 
                            [0, 2, 1, 3],
                            [0, 3, 1, 2],
                            [1, 2, 0, 3],
                            [1, 3, 0, 2],
                            [2, 3, 0, 1]])

        arrs = random_orderings([[0, 1], [2, 3]], [[0, 1, 2, 3]], 3)
        arrs = list(arrs)

        # Make sure we got three orderings
        self.assertEquals(len(arrs), 3)

        # And make sure they're all unique
        arrs = set(map(tuple, arrs))
        self.assertEquals(len(arrs), 3)


    def test_group_means(self):

        np.testing.assert_almost_equal(
            group_means(
                np.array([1, 2, 3, 4, 5]),
                [ [0, 1], [2, 3, 4] ]),
            np.array([1.5, 4.0]))

        np.testing.assert_almost_equal(
            group_means(
                np.array([[ 1,  2,  3,  4,  5],
                          [10, 20, 30, 40, 50]]),
                [ [0, 1], [2, 3, 4] ]),
            np.array([[ 1.5,   4.0],
                      [15.0,  40.0]]))


        data = np.array([6, 8, 4, 5, 3, 4,
                         8, 12, 9, 11, 6, 8,
                         13, 9, 11, 8, 7, 12])
        
        np.testing.assert_almost_equal(
            group_means(data, [ np.arange(0, 6),
                                np.arange(6, 12),
                                np.arange(12, 18) ]),
            np.array([5, 9, 10]))


    def test_group_rss(self):
        np.testing.assert_almost_equal(
            rss(
                np.array([1, 2, 3, 4, 5], float),
                [ [0, 1], [2, 3, 4] ]),
            2.5)

        np.testing.assert_almost_equal(
            rss(
                np.array([[ 1,  2,  3,  4,  5],
                          [10, 20, 30, 40, 50]], float),
                [ [0, 1], [2, 3, 4] ]),
            [2.5, 250])

        data = np.array([6, 8, 4, 5, 3, 4,
                         8, 12, 9, 11, 6, 8,
                         13, 9, 11, 8, 7, 12], float)

        np.testing.assert_almost_equal(
            rss(data, [ np.arange(0, 6),
                                np.arange(6, 12),
                                np.arange(12, 18) ]),
            np.array(68))

    def test_one_sample_t_test(self):
        row1 = np.array([1., 2., 3., 4.])
        row2 = np.array([-2, -3, 4, 5])
        table = np.vstack((row1, row2))
        
        test = OneSampleTTest()
        expected = np.array([4.47213595, 0.56568542])

        np.testing.assert_almost_equal(test(row1), expected[0])
        np.testing.assert_almost_equal(test(row2), expected[1])
        np.testing.assert_almost_equal(test(table), expected)
        
        test = OneSampleTTest(alphas=[0])
        np.testing.assert_almost_equal(test(row1), expected[0])
        np.testing.assert_almost_equal(test(row2), expected[1])
        np.testing.assert_almost_equal(test(table), [ expected ])


    def test_random_orderings_paired(self):

        for n in range(7):

            self.assertEquals(len(pairedOrderings(n, 100)), 2 ** n)

        self.assertEquals(len(pairedOrderings(10, 100)), 100)

    def test_layout_is_paired(self):
        self.assertTrue(layout_is_paired([[0,1], [2,3], [4,5]]))
        self.assertFalse(layout_is_paired([[0,1,2],[3,4,5]]))

    def test_means_ratio(self):
        condition_layout = [[0, 1, 2], [3, 4, 5]]
        block_layout     = [[0, 1, 2,   3, 4, 5]]

        row1 = np.array([0, 2, 7, 4, 6, 8])
        row2 = np.array([5, 4, 3, 2, 1, 0])
        table = np.vstack((row1, row2))

        # Asymmetric
        test = MeansRatio(condition_layout, block_layout, symmetric=False)
        np.testing.assert_almost_equal(test(row1), 0.5)
        np.testing.assert_almost_equal(test(row2), 4.0)
        np.testing.assert_almost_equal(test(table), [0.5, 4.0])

        # Symmetric
        test = MeansRatio(condition_layout, block_layout)
        np.testing.assert_almost_equal(test(row1), 2.0)
        np.testing.assert_almost_equal(test(row2), 4.0)
        np.testing.assert_almost_equal(test(table), [2.0, 4.0])


        alphas = np.array([0.0, 1.0, 2.0])

        # Asymmetric with alphas
        test = MeansRatio(condition_layout, block_layout, 
                          symmetric=False,
                          alphas=alphas)
        np.testing.assert_almost_equal(test(row1), [3. / 6., 4. / 7., 5. / 8.])
        np.testing.assert_almost_equal(test(row2), [4.0, 5. / 2., 2.])
        np.testing.assert_almost_equal(test(table), 
                                       np.array([[3. / 6.,  4. / 1.],
                                                 [4. / 7.,  5. / 2.],
                                                 [5. / 8.,  6. / 3.]]))

        # Symmetric with alphas
        test = MeansRatio(condition_layout, block_layout, 
                          alphas=alphas)
        np.testing.assert_almost_equal(test(row1), [6. / 3., 7. / 4., 8. / 5.])
        np.testing.assert_almost_equal(test(row2), [4.0, 5. / 2., 2.])
        np.testing.assert_almost_equal(test(table), 
                                       np.array([[6. / 3.,  4. / 1.],
                                                 [7. / 4.,  5. / 2.],
                                                 [8. / 5.,  6. / 3.]]))


    def test_multi_means_ratio(self):
        blocks     = [[0,   1], [2,   3], [4, 5]] # Pig
        conditions = [[0, 2, 4], [1, 3, 5]]
        
        row1 = np.array([0, 2, 7, 4, 6, 8])
        row2 = np.array([5, 4, 3, 2, 1, 0])
        table = np.vstack((row1, row2))

        tuning_param = 1

        test = MeansRatio(condition_layout=conditions,
                          block_layout=blocks,
                          symmetric=False, alphas=[1.])

        np.testing.assert_almost_equal(test(row1), [0.7457926])
        np.testing.assert_almost_equal(test(row2), [1.4736126])
        np.testing.assert_almost_equal(test(table), 
                                       np.array([[0.7457926, 1.4736126]]))


    def test_group_symbols(self):
        test = GroupSymbols([ [ 0, 1, 2, 3 ], [ 4, 5, 6, 7] ])
        self.assertEquals(test(np.array([0, 1, 2, 3, 4, 5, 6, 7])), 'AAAA BBBB')
        self.assertEquals(test(np.array([0, 1, 4, 5, 2, 3, 6, 7])), 'AABB AABB')
        self.assertEquals(test(np.array([0, 5, 6, 7, 4, 1, 2, 3])), 'ABBB AAAB')

        test = GroupSymbols([ [ 0, 1, 2 ], [ 3, 4, 5] ])
        self.assertEquals(test(np.array([0, 1, 2, 3, 4, 5])), 'AAA BBB')
        self.assertEquals(test(np.array([0, 1, 3, 2, 4, 5])), 'AAB ABB')

        test = GroupSymbols([ [ 0, 1 ], [ 2, 3] ])
        self.assertEquals(test(np.array([0, 1, 2, 3])), 'AA BB')
        self.assertEquals(test(np.array([0, 2, 0, 3])), 'AB AB')

if __name__ == '__main__':
    unittest.main()
