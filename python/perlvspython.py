#!/usr/bin/env python

import numpy as np
import argparse 
import unittest
import re
import subprocess
import StringIO
import time
import itertools

DIR = 'profile'
STDOUT_FILENAME = DIR + '/stdout'
STDERR_FILENAME = DIR + '/stderr'
STATS_FILENAME  = DIR + '/stats'

########################################################################
###
### Functions assigned to command-line actions
###

def simulate(args):
    in_filename = 'sample_data/2_class_testdata_header1.txt'
    header = ""
    lines  = []

    line_counts = np.array(10 ** log_ns(args), dtype=int)

    with open (in_filename) as f:
        header = f.next()

        for line in f:
            (rownum, rest) = line.split("\t", 1)
            lines.append(rest)
        
    for line_count in line_counts:
        make_sample_file(header, lines, line_count)

def run_page(args):

    with open(STATS_FILENAME, 'w') as stats:

        for t in range(args.times):
            for log_n in log_ns(args):
                n = int(10 ** log_n)

                filename = sample_filename(n)

                result = time_proc(['./page.py', '--channels', '1', filename])
                result['n'] = n
                result['version'] = 'python'
                stats.write(repr(result) + "\n")
                stats.flush()

                result = time_proc(['perl', 'PaGE_demo.pl', 
                                    '--infile', filename,
                                    '--num_channels', '1',
                                    '--unpaired', '--data_not_logged',
                                    '--level_confidence', 'L',
                                    '--min_presence', '4',
                                    '--tstat', '--use_unlogged_data'])
                result['n'] = n
                result['version'] = 'perl'
                stats.write(repr(result) + "\n")
                stats.flush()

def make_report(args):
    results = []
    with open(STATS_FILENAME) as f:
        results = [eval(line) for line in f]

    n_set       = set([x['n'] for x in results])
    version_set = set([x['version'] for x in results])

    make_index = lambda S: dict(map(None, S, range(len(S))))
    
    n_idx       = make_index(sorted(n_set))
    version_idx = make_index(version_set)

    times = int(np.ceil(float(len(results)) / (len(n_idx) * len(version_idx))))

    shape = (len(n_idx), len(version_idx), times)
    print "Shape is " + str(shape)
    real_times = np.zeros(shape)
    rss        = np.zeros(shape)

    for x in results:
        key = (n_idx[x['n']],
               version_idx[x['version']],
               0)
        print "Key is " + str(key)

        real_times[key] = x['real']
        rss[key] = x['maximum resident set size']

    real_times = np.mean(real_times, axis=2)
    rss        = np.mean(rss,        axis=2)

    perl = version_idx['perl']
    python = version_idx['python']

    ns = np.array(sorted(n_set))
    print ns
    print real_times[:, perl] / real_times[:, python]
    print rss[:, python] / rss[:, perl]

########################################################################
###
### Other things
###

def sample_filename(n):
    return 'profile/in_{0:d}.txt'.format(n)

def make_sample_file(header, lines, num_lines):
    filename = sample_filename(num_lines)
    print "Writing {0}".format(filename)

    n = len(lines)
    with open(filename, 'w') as out:
        out.write(header)
        for counter in range(num_lines):
            out.write("{0:d}\t{1:s}".format(counter, lines[counter % n]))
    out.close()


def log_ns(args):
    return np.linspace(3, args.max_log_n, (args.max_log_n - 3) * args.step + 1)

def time_proc(cmd):
    cmd = ['/usr/bin/time', '-l'] + cmd
    print "Running " + str(cmd)

    stderr = open(STDERR_FILENAME, 'w')
    stdout = open(STDOUT_FILENAME, 'a')
    subprocess.call(cmd, stderr=stderr, stdout=stdout)
    stderr.close()
    stderr = open(STDERR_FILENAME)
    result = parse_timing([line for line in stderr])
    stderr.close()
    return result
    
def main():
    parser = argparse.ArgumentParser()

    subs = parser.add_subparsers(help='Action to perform')

    sim_parser = subs.add_parser('simulate', help='Create simulated input files')
    sim_parser.add_argument('--max-log-n', type=int, default=8)
    sim_parser.add_argument('--step', type=int, default=2)
    sim_parser.set_defaults(func=simulate)

    run_parser = subs.add_parser('run', help='Run page')
    run_parser.add_argument('--max-log-n', type=int, default=8)
    run_parser.add_argument('--step', type=int, default=2)
    run_parser.add_argument('--times', type=int, default=1)
    run_parser.set_defaults(func=run_page)

    report_parser = subs.add_parser('report')
    report_parser.set_defaults(func=make_report)

    args = parser.parse_args()

    args.func(args)

def run_tests(args):
    unittest.main()
                
def parse_timing(lines):
    res = {}
    for (val, name) in re.findall("(\d+\.?\d*)\s+(real|user|sys)", lines[0]):
        res[name] = float(val)
    for line in lines[1:]:
        for (val, name) in re.findall("\s*(\d+)\s+(.*)", line):
            res[name] = float(val)
    return res

if __name__ == '__main__':
    main()

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
