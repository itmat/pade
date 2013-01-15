import sys
import page
import cProfile
import page.main

sys.argv = ['foo', 'run', '--directory', 'perf_report/new_1000']
cProfile.run('page.main.main()', 'run_prof')
