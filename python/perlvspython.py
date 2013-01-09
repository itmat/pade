#!/usr/bin/env python

import matplotlib
matplotlib.use('pdf')

import numpy as np
import argparse 
import unittest
import re
import subprocess
import StringIO
import time
import itertools
import matplotlib.pyplot as plt
import os
import shutil

DIR = 'profile'
STDOUT_FILENAME = DIR + '/stdout'
STDERR_FILENAME = DIR + '/stderr'
STATS_FILENAME  = DIR + '/stats'
REPORT_FILENAME = 'report.tex'

########################################################################
###
### Functions assigned to command-line actions
###

def setup(args):
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
                directory = os.path.join('profile', 'new_{0}'.format(n))
                result = time_proc(['./page.py', 'run', '--directory', directory])
                result['n'] = n
                result['version'] = 'python'
                stats.write(repr(result) + "\n")
                stats.flush()

#                result = time_proc(['perl', 'PaGE_demo.pl', 
#                                    '--infile', filename,
#                                    '--num_channels', '1',
#                                    '--unpaired', '--data_not_logged',
#                                    '--level_confidence', 'L',
#                                    '--min_presence', '4',
#                                    '--tstat', '--use_unlogged_data'])
#                result['n'] = n
#                result['version'] = 'perl'
#                stats.write(repr(result) + "\n")
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

#    perl = version_idx['perl']
    python = version_idx['python']

    ns = np.array(sorted(n_set))
    versions = np.array(version_set)

    suffix = '(2 classes, 4 reps per class)'

    # Plot running times
#    perl_minutes   = real_times[:, perl] / 60.0
    python_minutes = real_times[:, python] / 60.0
#    plt.plot(ns, perl_minutes, label='perl')
    plt.plot(ns, python_minutes, label='python')
    plt.title('Running times ' + suffix)
    plt.xlabel('Features')
    plt.semilogx()
    plt.ylabel('Minutes')
    plt.savefig('runningtime')

    # Plot memory usage
    plt.clf()
#    perl_gigs   = rss[:, perl] / 1000000000.0
    python_gigs = rss[:, python] / 1000000000.0
#    plt.plot(ns, perl_gigs, label='perl')
    plt.plot(ns, python_gigs, label='python')
    plt.xlabel('Features')
    plt.ylabel('Memory (GB)')
    plt.semilogx()
    plt.title('Memory usage ' + suffix)
    plt.savefig('rss')

    # Plot improvement of running time and memory usage

#    rss_improvement = rss[:, perl] / rss[:,python]
#    time_improvement = real_times[:, perl] / real_times[:,python]
#    plt.clf()
#    plt.plot(ns, rss_improvement, label='Memory usage')
#    plt.plot(ns, time_improvement, label='Running time')
#    plt.text(ns[3], rss_improvement[3] - 1, 'Memory usage')
#    plt.text(ns[3], time_improvement[3] - 1, 'Running time')
#    plt.xlabel('Features')
#    plt.ylabel('perl / python')
#    plt.semilogx()
#    plt.title('Improvement (perl / python)')
#    plt.savefig('improvement')

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

    directory = 'profile/new_{0}'.format(num_lines)
    subprocess.call(['python', 'page.py', 
                     'setup', 
                     '--factor', 'treated', 
                     '--directory', directory,
                     filename
                     ])

    shutil.copy('sample_data/schema_2class.yaml', os.path.join(directory, 'schema.yaml'))

def log_ns(args):
    return np.linspace(3, args.max_log_n, (args.max_log_n - 3) * args.step + 1)

def time_proc(cmd):
    cmd = ['/usr/bin/time', '-l'] + cmd
    print "Running " + str(cmd)

    stderr = open(STDERR_FILENAME, 'w')
    stdout = open(STDOUT_FILENAME, 'a')
    retcode = subprocess.call(cmd, stderr=stderr, stdout=stdout)
    if retcode != 0:
        raise Exception("Call failed")
    stderr.close()
    stderr = open(STDERR_FILENAME)
    result = parse_timing([line for line in stderr])
    stderr.close()
    return result
    
def main():
    parser = argparse.ArgumentParser()

    if not os.path.exists('profile'):
        os.makedirs('profile')

    subs = parser.add_subparsers(help='Action to perform')

    sim_parser = subs.add_parser('setup', help='Setup jobs for profiling')
    sim_parser.add_argument('--max-log-n', type=int, default=5)
    sim_parser.add_argument('--step', type=int, default=2)
    sim_parser.set_defaults(func=setup)

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

    parts = lines[0].strip().split()
    print "Parts are", parts
    for i in range(3):
        res[parts[i * 2 + 1]] = float(parts[i * 2])
    for line in lines[1:]:
        for (val, name) in re.findall("\s*(\d+)\s+(.*)", line):
            res[name] = float(val)
    return res

if __name__ == '__main__':
    main()

