import unittest
from page import *

class ProfileTest(unittest.TestCase):

    def setUp(self):
        # def foo():
        #     print 1 + 2 * 3
        # def main():
        #     foo()
        #     foo()
        self.profile_log = [
            ('enter', 'top',  1),
            ('enter', 'foo',  2),
            ('exit',  'foo',  3),
            ('enter', 'bar',  5),
            ('enter', 'baz',  7),
            ('exit',  'baz', 11),
            ('exit',  'bar', 13),
            ('exit',  'top', 17)]

    def test_walk_profile(self):
        scanned = walk_profile(self.profile_log)
        print scanned
        expected_maxrss_post = [17., 3., 13., 11.]
        
        self.assertTrue(np.all(scanned['order'] == [0, 1, 2, 3]))
        self.assertTrue(np.all(scanned['depth'] == [0, 1, 1, 2]))
        np.testing.assert_almost_equal(scanned['maxrss_pre'], [1., 2., 5., 7.])
        np.testing.assert_almost_equal(scanned['maxrss_post'], [17., 3., 13., 11.])
