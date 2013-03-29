import pade.tasks
import statsmodels.api as sm
import pade.analysis as an
from pade.stat import GLMFStat, FStat
import numpy as np


if __name__ == '__main__':

    job = pade.tasks.load_job("padedb/job_7.pade")
    f = GLMFStat(job.condition_layout, job.block_layout)                         

    idxs = [717]
    Y = job.input.table


    glmf  = GLMFStat(job.condition_layout, job.block_layout)
    f     = FStat(job.condition_layout, job.block_layout)


    for i in idxs:
        print f(Y[i]), glmf(Y[i])



    cond_layout = [range(0, 8), range(8, 16), range(16, 24)]
    block_layout = [ range(24) ]

    glmf  = GLMFStat(cond_layout, block_layout)
    f     = FStat(cond_layout, block_layout)

    for i in idxs:
        print f(Y[i]), glmf(Y[i])

