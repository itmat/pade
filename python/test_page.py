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

        self.assertAlmostEqual(alphas[1], 1.62026604316528)
        self.assertAlmostEqual(alphas[2], 1.61770701155527)
        self.assertAlmostEqual(alphas[3], 1.60540468969643)

        page.compute_s(data.table[:,(0,1,2,3)],
                         data.table[:,(4,5,6,7)], 10, 10)

    def test_vectorized_tstat(self):
        v1 = [[2.410962, 1.897421, 2.421239, 1.798668],
              [2.410962, 1.897421, 2.421239, 1.798668]]
        v2 = [[0.90775,  0.964438, 1.07578,  1.065872],
              [0.90775,  0.964438, 1.07578,  1.065872]]
        v1 = array(v1)
        v2 = array(v2)
        alpha = 1.62026604316528
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

        self.assertAlmostEqual(sum(page.v_tstat(v1, v2, alpha, axis=1)),
                               sum(expected),
                               )


    def test_min_max_tstat(self):
        data = page.load_input(self.config)
        alphas = page.find_default_alpha(data)
        (mins, maxes) = page.min_max_stat(data, alphas)
        
        e_mins = array([
                [0.0, -4.92023590278385, -43.8283558215159, -6.07518670089633],
                [0.0, -3.89533331728057, -39.6484589883064, -5.88208044481599],
                [0.0, -3.52368811319764, -21.2364854229474, -4.56341648537748],
                [0.0, -3.26242875610601, -10.4512484892809, -3.8301409378307],
                [0.0, -3.03723671683252, -6.93116328105236, -3.63883036028571],
                [0.0, -2.59025081540864, -3.76279534584871, -3.23488462096825],
                [0.0, -2.25795126899826, -2.58235242575245, -2.91166170987209],
                [0.0, -2.00121790614259, -2.37354145548531, -2.64716274082434],
                [0.0, -1.63044784992463, -2.1602747732769, -2.24016414899138],
                [0.0, -0.742246110072031, -1.32616667028968, -1.07895089421107],
                ])
        e_maxes = array([
                [0.0, 33.8513438763702, 40.1569352610503, 32.1406090722972],
                [0.0, 32.6237150390861, 39.4853265500449, 30.2063064916352],
                [0.0, 24.5349298465585, 34.2742126910379, 19.5242958951605],
                [0.0, 15.8189654039207, 26.5017746150241, 16.1720091769614],
                [0.0, 13.9061831087407, 21.6028492941717, 14.2556982740597],
                [0.0, 11.0639344349816, 14.7748985540123, 10.997740402287],
                [0.0, 9.18635846949678, 11.2265538223892, 8.95189831100723],
                [0.0, 7.85358622430092, 9.05250185522534, 7.5478231368327],
                [0.0, 6.08728054656708, 6.52524125360866, 5.74549877705118],
                [0.0, 2.36460731135669, 2.30890874240908, 2.75756181693387]
                ])
        
        self.assertTrue(all(abs(mins  - e_mins)  < 0.00001))
        self.assertTrue(all(abs(maxes - e_maxes) < 0.00001))


    def test_all_subsets(self):
        subsets = page.all_subsets(8, 4)
        # There should be 70 rows and 8 columns
        self.assertEquals(shape(subsets), (70, 4))
        # There should be 4 1s and 4 0s in each row
#        self.assertTrue(all(sum(subsets, axis=1) == 4))



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
