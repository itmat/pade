import sys
import page.ui
import cProfile

sys.argv = ['foo', 'run', 'sample_data/4_class_testdata_header1.txt']
cProfile.run('page.ui.main()', 'run_prof')
