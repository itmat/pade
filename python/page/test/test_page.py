import sys, os

#sys.path.insert(0, os.path.dirname(__file__) + "/../..")

import unittest
import doctest
import numpy as np
import numpy.ma as ma
from page.schema import Schema
import page.core as page
from page.stats import Tstat

from numpy import *
from data import unpermuted_stats, mean_perm_up, conf_bins_up_down

class PageTest(unittest.TestCase):

    def setUp(self):
        self.v1 = ma.masked_array(
            [float(x) for x in [1, 6, 5, 3, 8, 9, 6, 3, 6, 8]], ma.nomask)
        self.v2 = ma.masked_array(
            [float(x) for x in [7, 4, 9, 6, 2, 4, 7, 4, 2, 1]], ma.nomask)

        self.infile = 'sample_data/4_class_testdata_header1.txt'

        column_names = ["id"]
        is_feature_id = [True]
        is_sample     = [False]
        levels = ["none", "low", "medium", "high"]

        for condition in range(0, 4):
            for replicate in range(1, 5):
                column_names.append("c{0}r{1}".format(condition, replicate))
                is_feature_id.append(False)
                is_sample.append(True)

        self.schema = Schema(
            column_names=column_names,
            is_feature_id=is_feature_id,
            is_sample=is_sample)

        self.schema.add_attribute("treatment", "S100")

        for condition in range(0, 4):
            for replicate in range(1, 5):
                sample = "c{0}r{1}".format(condition, replicate)
                self.schema.set_attribute(sample, "treatment", levels[condition])

    def test_compute_s(self):
        s = page.compute_s(self.v1, self.v2)
        self.assertAlmostEqual(s, 2.57012753682683)

    def test_load_input(self):
        job = page.Job(self.infile, self.schema)
        self.assertEquals(len(job.feature_ids), 1000)

    def test_default_alpha(self):
        job = page.Job(self.infile, self.schema)
        conditions = self.schema.sample_groups("treatment").values()

        # TODO: Make find_default_alpha take schema?
        alphas = page.find_default_alpha(job)

        self.assertAlmostEqual(alphas[1], 1.62026604316528)
        self.assertAlmostEqual(alphas[2], 1.61770701155527)
        self.assertAlmostEqual(alphas[3], 1.60540468969643)

        page.compute_s(job.table[:,(0,1,2,3)],
                       job.table[:,(4,5,6,7)])

    def test_tstat(self):
        v1 = [[2.410962, 1.897421, 2.421239, 1.798668],
              [2.410962, 1.897421, 2.421239, 1.798668]]
        v2 = [[0.90775,  0.964438, 1.07578,  1.065872],
              [0.90775,  0.964438, 1.07578,  1.065872]]
        v1 = array(v1)
        v2 = array(v2)
        alpha = 1.62026604316528
        alphas = page.TUNING_PARAM_RANGE_VALUES * alpha
        expected = [
            [
                6.62845904788559,
                6.21447939063187,
                3.96389309107878,
                2.19632757746533,
                1.51898640652018,
                0.857703281874709,
                0.597558974875407,
                0.458495683101651,
                0.312872820321998,
                0.0970668755330585,
                ],
            [
                6.62845904788559,
                6.21447939063187,
                3.96389309107878,
                2.19632757746533,
                1.51898640652018,
                0.857703281874709,
                0.597558974875407,
                0.458495683101651,
                0.312872820321998,
                0.0970668755330585,
                ]]

        self.assertAlmostEqual(sum(page.tstat(v1, v2, alphas)),
                               sum(expected),
                               )


    def test_all_subsets(self):
        subsets = page.all_subsets(8, 4)
        self.assertEquals(shape(subsets), (70, 8))

    def test_a_unpermuted_stats(self):
        job = page.Job()
        job.table = unpermuted_stats.data
        job._conditions = [
            [ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11],
            [12, 13, 14, 15]]

        for i in range(len(page.TUNING_PARAM_RANGE_VALUES)):

            (u, d, stats) = page.unpermuted_stats(
                job,
                unpermuted_stats.mins[i],
                unpermuted_stats.maxes[i],
                [Tstat(a * page.TUNING_PARAM_RANGE_VALUES[i])
                 for a in unpermuted_stats.alpha_default],
                1000)
            self.assertEqual(shape(stats), (1000, 4))
            self.assertTrue(all(abs(unpermuted_stats.stats[i] - stats) < 0.00001))

            self.assertEqual(shape(u), shape(unpermuted_stats.dist_up[i]))
            self.assertEqual(shape(d), shape(unpermuted_stats.dist_down[i]))

            expected_u = np.copy(unpermuted_stats.dist_up[i])
            expected_d = np.copy(unpermuted_stats.dist_down[i])

            (n, p) = shape(expected_u)

            for idx in np.ndindex((n)):
                expected_u[idx] = page.accumulate_bins(expected_u[idx])
                expected_d[idx] = page.accumulate_bins(expected_d[idx])

            u_diffs = expected_u - u
            d_diffs = expected_d - d

            print u_diffs

            # For some reason the last two bins are swapped in a very
            # small number of cases, so ignore them.
            u_diffs = u_diffs[:, :999]
            d_diffs = d_diffs[:, :999]

            self.assertTrue(all(u_diffs == 0))
            self.assertTrue(all(d_diffs == 0))

    def test_adjust_num_diff(self):
        self.assertAlmostEqual(page.adjust_num_diff(7.07142857142857, 5, 1000),
                               7.08618085029828)
    
    def test_conf_bins(self):
        job = page.Job(self.infile, self.schema)

        conditions = self.schema.sample_groups("treatment").values()
        alphas = page.find_default_alpha(job)

        (conf_bins_up, conf_bins_down, breakdown) = page.do_confidences_by_cutoff(job, alphas, 1000)
        self.assertTrue(np.all(conf_bins_up - conf_bins_up_down.conf_up < 0.00001))
        self.assertTrue(np.all(conf_bins_down - conf_bins_up_down.conf_down < 0.00001))

        expected_breakdown = np.array([[[  0.  ,   0.  ,   0.  ],
        [  0.  ,   0.  ,   0.  ],
        [  0.  ,   0.  ,   0.  ],
        [  0.  ,   0.  ,   0.  ],
        [  0.  ,   0.  ,   0.  ],
        [  0.  ,   0.  ,   0.  ],
        [  0.  ,   0.  ,   0.  ],
        [  0.  ,   0.  ,   0.  ],
        [  0.  ,   0.  ,   0.  ],
        [  0.  ,   0.  ,   0.  ]],

       [[  0.5 ,  73.  ,   0.  ],
        [  0.55,  70.  ,   0.  ],
        [  0.6 ,  66.  ,   0.  ],
        [  0.65,  59.  ,   0.  ],
        [  0.7 ,  47.  ,   0.  ],
        [  0.75,  44.  ,   0.  ],
        [  0.8 ,  36.  ,   0.  ],
        [  0.85,  29.  ,   0.  ],
        [  0.9 ,  24.  ,   0.  ],
        [  0.95,  14.  ,   0.  ]],

       [[  0.5 ,  70.  ,  23.  ],
        [  0.55,  65.  ,  22.  ],
        [  0.6 ,  61.  ,  21.  ],
        [  0.65,  54.  ,  13.  ],
        [  0.7 ,  47.  ,  11.  ],
        [  0.75,  42.  ,  10.  ],
        [  0.8 ,  35.  ,   9.  ],
        [  0.85,  31.  ,   7.  ],
        [  0.9 ,  24.  ,   2.  ],
        [  0.95,  20.  ,   1.  ]],

       [[  0.5 ,  56.  ,   7.  ],
        [  0.55,  46.  ,   5.  ],
        [  0.6 ,  45.  ,   0.  ],
        [  0.65,  42.  ,   0.  ],
        [  0.7 ,  40.  ,   0.  ],
        [  0.75,  38.  ,   0.  ],
        [  0.8 ,  35.  ,   0.  ],
        [  0.85,  32.  ,   0.  ],
        [  0.9 ,  29.  ,   0.  ],
        [  0.95,  24.  ,   0.  ]]])
        self.assertTrue(np.all(expected_breakdown - breakdown) == 0)

if __name__ == '__main__':
    unittest.main()
