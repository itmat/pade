import sys
import cProfile
import pstats

from pade.tasks import load_job

if __name__ == '__main__':

    path = sys.argv[1]
    job = load_job(path)
    table = job.input.table
    stat_fn = job.get_stat_fn()

    cProfile.run('stat_fn(table)', sort='time', filename='prof')

    p = pstats.Stats('prof')

    p.strip_dirs().sort_stats('time').print_stats(100)
