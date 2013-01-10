from perlvspython import *

import unittest

class TestPerlVsPython(unittest.TestCase):

    def setUp(self):
        self.cluster_time_out = """
        Command being timed: "find src/rum -name foo"
        User time (seconds): 0.05
        System time (seconds): 0.35
        Percent of CPU this job got: 4%
        Elapsed (wall clock) time (h:mm:ss or m:ss): 0:08.32
        Average shared text size (kbytes): 0
        Average unshared data size (kbytes): 0
        Average stack size (kbytes): 0
        Average total size (kbytes): 0
        Maximum resident set size (kbytes): 5152
        Average resident set size (kbytes): 0
        Major (requiring I/O) page faults: 0
        Minor (reclaiming a frame) page faults: 366
        Voluntary context switches: 2787
        Involuntary context switches: 140
        Swaps: 0
        File system inputs: 0
        File system outputs: 0
        Socket messages sent: 0
        Socket messages received: 0
        Signals delivered: 0
        Page size (bytes): 4096
        Exit status: 0
"""


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
