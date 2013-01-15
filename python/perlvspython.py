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
        make_sample_file(args.directory, header, lines, line_count)


def log_time(job, fh, directory):
    result = time_proc(job['cmd'], directory)
    for k in job:
        result[k] = job[k]
    fh.write(repr(result) + "\n")
    fh.flush()

def run_page(args):

    filename = args.outfile

    with open(filename, 'w') as stats:

        for t in range(args.times):
            for log_n in log_ns(args):

                n = int(10 ** log_n)

                filename = sample_filename(args.directory, n)
                directory = os.path.join('profile', 'new_{0}'.format(n))

                python_job = {
                    'n'       : n,
                    'version' : 'python',
                    'cmd'     : ['./page.py', 'run', '--directory', directory]
                    }

                perl_job = {
                    'n' : n,
                    'version' : 'perl',
                    'cmd' : ['perl', 'PaGE_demo.pl', 
                             '--infile', filename,
                             '--num_channels', '1',
                             '--unpaired', '--data_not_logged',
                             '--level_confidence', 'L',
                             '--min_presence', '4',
                             '--tstat', '--use_unlogged_data']
                    }

                log_time(python_job, stats, args.directory)
                log_time(perl_job, stats, args.directory)

def make_report(args):
    results = []

    for f in args.infiles:
        for line in f:
            try:
                results.append(eval(line))
            except Exception as e:
                raise Exception("Error eval'ing " + line + " from " + f, e)

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

        real_times[key] = x['elapsed_seconds']
        rss[key] = x['rss_gb']

    real_times = np.mean(real_times, axis=2)
    rss        = np.mean(rss,        axis=2)

    python = version_idx['python']
    perl   = version_idx['perl']

    ns = np.array(sorted(n_set))
    versions = np.array(version_set)

    suffix = '(2 classes, 4 reps per class)'

    # Plot running times
    python_minutes = real_times[:, python] / 60.0
    perl_minutes   = real_times[:, perl] / 60.0
    idxs = perl_minutes > 0


    plt.plot(ns, python_minutes, label='python')
    plt.plot(ns[idxs], perl_minutes[idxs], label='perl')
    plt.legend(loc=2)
    plt.title('Running times ' + suffix)
    plt.xlabel('Features')
    plt.semilogx()
    plt.ylabel('Minutes')
    plt.savefig(os.path.join(args.directory, 'runningtime'))

    # Plot memory usage
    plt.clf()
    python_gigs = rss[:, python]
    perl_gigs   = rss[:, perl]
    plt.plot(ns, python_gigs, label='python')
    plt.plot(ns[idxs], perl_gigs[idxs], label='perl')
    plt.xlabel('Features')
    plt.ylabel('Memory (GB)')
    plt.legend(loc=2)
    plt.semilogx()
    plt.title('Memory usage ' + suffix)
    plt.savefig(os.path.join(args.directory, 'rss'))



    # Plot improvement of running time and memory usage

    rss_improvement = rss[:, perl] / rss[:,python]
    time_improvement = real_times[:, perl] / real_times[:,python]
    plt.clf()
    plt.plot(ns[idxs], rss_improvement[idxs], label='Memory usage')
    plt.plot(ns[idxs], time_improvement[idxs], label='Running time')
    plt.xlabel('Features')
    plt.ylabel('perl / python')
    plt.legend(loc=2)
    plt.semilogx()
    plt.title('Improvement (perl / python)')
    plt.savefig(os.path.join(args.directory, 'improvement'))

########################################################################
###
### Other things
###

def sample_filename(directory, n):
    return os.path.join(directory, 'in_{0:d}.txt'.format(n))

def make_sample_file(directory, header, lines, num_lines):
    filename = sample_filename(directory, num_lines)
    print "Writing {0}".format(filename)

    n = len(lines)
    with open(filename, 'w') as out:
        out.write(header)
        for counter in range(num_lines):
            out.write("{0:d}\t{1:s}".format(counter, lines[counter % n]))
    out.close()

    directory = os.path.join(directory, 'new_{0}'.format(num_lines))

    retcode = subprocess.call(['python', 'page.py', 
                     'setup', 
                     '--factor', 'treated', 
                     '--directory', directory,
                     filename
                     ])
    if retcode != 0:
        raise Exception("Setup failed")
    shutil.copy('sample_data/schema_2class.yaml', os.path.join(directory, 'schema.yaml'))

def log_ns(args):
    return np.linspace(3, args.max_log_n, (args.max_log_n - 3) * args.step + 1)

def time_proc(cmd, directory):
    cmd = ['/usr/bin/time', '-f', '%e %M'] + cmd
    print "Running " + str(cmd)

    stderr_filename = os.path.join(directory, "stderr")
    stdout_filename = os.path.join(directory, "stdout")

    stderr = open(stderr_filename, 'w')
    stdout = open(stdout_filename, 'a')
    retcode = subprocess.call(cmd, stderr=stderr, stdout=stdout)
    if retcode != 0:
        raise Exception("Call failed: ")
    stderr.close()
    stderr = open(stderr_filename)
    lines = [x for x in stderr]
    print lines
    (elapsed_seconds, rss_kb) = lines[0].split()
    elapsed_seconds = float(elapsed_seconds)
    rss_gb = float(rss_kb) / 1000000
    result = {
        'elapsed_seconds' : elapsed_seconds,
        'rss_gb'          : rss_gb
        }
    return result
    
def main():
    parser = argparse.ArgumentParser()

    subs = parser.add_subparsers(help='Action to perform')

    sim_parser = subs.add_parser('setup', help='Setup jobs for profiling')
    sim_parser.add_argument('--max-log-n', type=int, default=5)
    sim_parser.add_argument('--step', type=int, default=2)
    sim_parser.add_argument('--directory', '-d', default='perf_report')
    sim_parser.set_defaults(func=setup)


    run_parser = subs.add_parser('run', help='Run page')
    run_parser.add_argument('--max-log-n', type=int, default=8)
    run_parser.add_argument('--step', type=int, default=2)
    run_parser.add_argument('--times', type=int, default=1)
    run_parser.add_argument('--directory', '-d', default='perf_report')
    run_parser.add_argument('--outfile', '-o')
    run_parser.set_defaults(func=run_page)

    report_parser = subs.add_parser('report')
    report_parser.add_argument('--directory', '-d', default='perf_report')
    report_parser.add_argument('infiles', nargs='+', type=file)
    report_parser.set_defaults(func=make_report)

    args = parser.parse_args()

    args.func(args)

def run_tests(args):
    unittest.main()
    
time_out_handlers = {
    'time' : {
        'User time (seconds)'
        }
}
            
def parse_timing(lines):
    res = {}

    for line in lines[1:]:
        for (name, val) in re.findall("\s*(.+):\s+(.*)", line):
            res[name] = float(val)
    return res

if __name__ == '__main__':
    main()

