import sys
import page
import cProfile

sys.argv = ['foo', 'run', '--directory', 'profile/new_1000']
cProfile.run('page.main()', 'run_prof')
