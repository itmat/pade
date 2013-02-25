import sys
import pade
import cProfile
import pade.main

if __name__ == '__main__':
    sys.argv = ['foo', 'run', '--directory', 'perf_report/new_1000']
    cProfile.run('pade.main.main()', 'run_prof')
