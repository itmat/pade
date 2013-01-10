from perlvspython import *

import unittest

class TestPerlVsPython(unittest.TestCase):

    def test_parse_time(self):

        time_output = """   4.13 real         4.03 user         0.08 sys
  38846464  maximum resident set size
         0  average shared memory size
         0  average unshared data size
         0  average unshared stack size
     10433  page reclaims
         0  page faults
         0  swaps
         0  block input operations
         0  block output operations
         0  messages sent
         0  messages received
         0  signals received
         0  voluntary context switches
        66  involuntary context switches
"""
        
        parsed = parse_timing(time_output.splitlines())
        self.assertAlmostEquals(parsed['real'], 4.13)
        self.assertAlmostEquals(parsed['sys'],  0.08)
        self.assertEquals(parsed['maximum resident set size'], 38846464)

if __name__ == '__main__':
    unittest.main()
