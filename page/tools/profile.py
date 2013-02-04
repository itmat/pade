import sys
import page
import cProfile
import page.main

if __name__ == '__main__':
    sys.argv = ['foo', 'run', '--directory', 'perf_report/new_1000']
    cProfile.run('page.main.main()', 'run_prof')
