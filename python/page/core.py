#!/usr/bin/env python



"""
M is the number of conditions
M_i is the number of replicates in the ith condition
N is the number of features


H is the number of bins
R is the number of permutations
T is the number of tests
L is the number of confidence levels



Plot:

* Histogram of feature count by statistic, one for each condition /
  alpha.

* Cumulative num features by statistic. One chart for each
  condition, with different alphas.

* Num features for each confidence level, one for each condition, with
  different alphas. Overlay "best" alpha at each confidence level.

Document:

* Differentiate between statistic bins and confidence bins
* Allow user-specified statistic bins?

"""
import matplotlib
import matplotlib.pyplot as plt
import collections
import os
import re
import itertools 
import logging
import numpy as np
import logging
import stats

from report import Report

class IntermediateResults:
    """Holds the intermediate results of a job.

    alphas - TODO
    
    stats - a T x C x N array, where T is the number of tests, C is
            the number of conditions, and N is the number of features.

    conf_levels - an array of floats of length L, representing the
                  (lower) edges of the confidence levels.

    up - A DirectionalResults object representing up-regulated features.

    down - A DirectionalResults object representing down-regulated
           features.

    """
    def __init__(self, alphas, stats, conf_levels, up, down):
        self.alphas = alphas
        self.stats  = stats
        self.conf_levels = conf_levels
        self.up = up
        self.down = down

    def save(self, output_dir):
        cwd = os.getcwd()
        try:
            os.chdir(output_dir)
            np.save('alphas', self.alphas)
            np.save('stats', self.stats)
            np.save('conf_levels', self.conf_levels)
            np.save('up_unperm_counts', self.up.unperm_counts)
            np.save('up_raw_conf', self.up.raw_conf)
            np.save('up_conf_to_stat', self.up.conf_to_stat)
            np.save('up_conf_to_count', self.up.conf_to_count)
            np.save('up_best_params', self.up.best_params)
            np.save('down_unperm_counts', self.down.unperm_counts)
            np.save('down_raw_conf', self.down.raw_conf)
            np.save('down_conf_to_stat', self.down.conf_to_stat)
            np.save('down_conf_to_count', self.down.conf_to_count)
            np.save('down_best_params', self.down.best_params)

        finally:
            os.chdir(cwd)

    @classmethod
    def load(cls, dir_):
        """Load saved results from the given directory."""

        cwd = os.getcwd()
        try:
            
            os.chdir(dir_)
            alphas = np.load('alphas.npy')
            stats = np.load('stats.npy')
            conf_levels = np.load('conf_levels.npy')
            up = DirectionalResults(
                np.load('up_unperm_counts.npy'),
                np.load('up_raw_conf.npy'),
                np.load('up_conf_to_stat.npy'),
                np.load('up_conf_to_count.npy'),
                np.load('up_best_params.npy'))
            down = DirectionalResults(
                np.load('down_unperm_counts.npy'),
                np.load('down_raw_conf.npy'),
                np.load('down_conf_to_stat.npy'),
                np.load('down_conf_to_count.npy'),
                np.load('down_best_params.npy'))

            return IntermediateResults(alphas, stats, conf_levels, up, down)
        
        finally:
            os.chdir(cwd)


    def directional(self, direction):
        if direction == 'up':
            return self.up
        if direction == 'down':
            return self.down
        raise Exception("Unknown direction " + direction)

    @property
    def best_up_stats_by_level(self):
        return self.best_stats_by_level('up')

    @property
    def best_down_stats_by_level(self):
        return self.best_stats_by_level('down')

    def cutoffs_by_level(self, direction):
        """Returns a level x class array. 

        cutoffs_by_level[level, class] gives the value of the
        statistic to use for the given convidence level and class.

        """
        directional = self.directional(direction)
        params = directional.best_params
        C = np.shape(self.stats)[1]
        L = len(self.conf_levels)
        print "Shape of conf_to_stat is " + str(np.shape(directional.conf_to_stat))

        res = np.zeros((L, C))
        for l in range(L):
            for c in range(C):
                param = params[c, l]
                cutoff = directional.conf_to_stat[param, c, l]
                print "Cutoff is " + str(cutoff)
                res[l, c] = cutoff
        print "Cutoffs for " + direction + " are " + str(res)
        return res

    @property
    def up_cutoffs_by_level(self):
        return self.cutoffs_by_level('up')

    def best_stats_by_level(self, direction):
        """Returns a level x class x feature array.

        The result gives the statistic for the given feature in the
        given class, using the value of alpha that maximizes the power
        for the given level.

        """

        directional = self.directional(direction)

        C = np.shape(self.stats)[1]
        N = np.shape(self.stats)[2]
        L = len(self.conf_levels)
        res = np.zeros((L, C, N))

        for i in range(L):
            params = directional.best_params[:, i]
            print "Params are " + str(params)
            for c in range(C):
                stats = self.stats[params[c], c]
                res[i, c] = stats
        return res
        
class DirectionalResults:
    """Data describing up- or down- regulated features.

    unperm_counts - For each statistic and condition, gives the
                    distribution over all the features. A T x C x B
                    array, where T is the number of tests, C is the
                    number of classes, and B is the number of bins
                    used to discretize the statistic space.

    raw_conf - Gives the confidence level (basically 1 - FDR) for each
               bin represented by unperm_counts.

    conf_to_count - A T x C x L array, where T is the number of tests,
                    C is the number of conditions, and L is the number
                    of confidence levels. conf_to_count[t, c, l] gives
                    the number of features in that test t shows as up-
                    or down- regulated for condition c in confidence
                    level l.

    """


    def __init__(self, unperm_counts, raw_conf, conf_to_stat, conf_to_count, best_params):
        self.unperm_counts = unperm_counts
        self.raw_conf = raw_conf
        self.conf_to_stat = conf_to_stat
        self.conf_to_count = conf_to_count
        self.best_params = best_params

    @property
    def best_counts(self):
        # Num tests, num conf levels, num conditions
        (S, N, L) = np.shape(self.conf_to_count)
                
        res = np.zeros((N, L))

        for c in range(N):
            for l in range(L):
                test_idx = self.best_params[c, l]
                res[c, l] = self.conf_to_count[test_idx, c, l]
        return res


class Job(object):

    levels = np.linspace(0.5, 0.95, 10)

    def __init__(self, fh=None, schema=None):
        self.infile = fh
        self.schema = schema

        """Read input from the given filehandle and return the
    data, the feature ids, and the layout of the conditions and
    replicates.

    Returns a tuple of three items: (table, row_ids,
    conditions). Table is an (m x n) array of floats, where m is the
    number of features and n is the total number of replicates for all
    conditions. row_ids is an array of length m, where the ith entry
    is the name of the ith feature in the table. conditions is a list
    of lists. The ith list gives the column indices for the replicates
    of the ith condition. For example:

    [[0,1],
     [2,3,4],
     [5,6,7,8]]

    indicates that there are three conditions. The first has two
    replicates, in columns 0 and 1; the second has three replicates,
    in columns 2, 3, and 4; the third has four replicates, in columns
    5 through 8.
    """

        if type(fh) == str:
            fh = open(fh, 'r')

        self.feature_ids = None
        self.table = None
        self._conditions = None
        self._condition_names = None
        
        if fh is not None:
            headers = fh.next().rstrip().split("\t")

            ids = []
            table = []

            for line in fh:
                row = line.rstrip().split("\t")
                rowid = row[0]
                values = [float(x) for x in row[1:]]
                ids.append(rowid)
                table.append(values)

            table = np.array(table)

            self.table = table
            self.feature_ids   = ids


    @property
    def conditions(self):
        """A list of lists of indices into self.table. Each inner list
        is a list of indices for samples that are in the same
        condition."""

        if self._conditions is None:
            groups = self.schema.sample_groups(self.schema.attribute_names[0])
            self._conditions = groups.values()
        return self._conditions

    @property
    def condition_names(self):
        if self._condition_names is None:
            groups = self.schema.sample_groups(self.schema.attribute_names[0])
            self._condition_names = groups.keys()
        return self._condition_names

########################################################################
###
### Constants
###

__version__ = '6.0.0'


########################################################################
###
### Functions
###

################################################################
###
### Low-level functions
###

def compute_s(v1, v2, axis=0):
    """
    v1 and v2 should have the same number of rows.
    """

    var1 = np.var(v1, ddof=1, axis=axis)
    var2 = np.var(v2, ddof=1, axis=axis)
    
    s1 = np.size(v1, axis=axis) - 1
    s2 = np.size(v2, axis=axis) - 1

    return np.sqrt((var1 * s1 + var2 * s2)
                   / (s1 + s2))


def summarize_confidence(levels, unperm_counts, raw_conf, bins):
    """
    Return the lowest statistic at which the confidence level is greater or equal to conf.

    """

    base_shape = np.shape(unperm_counts)[:-1]

    conf_to_stat  = np.zeros(base_shape + (len(levels),))
    conf_to_count = np.zeros(base_shape + (len(levels),), int)

    logging.debug("Shape of raw conf is " + str(np.shape(raw_conf)))

    for idx in np.ndindex(base_shape):

        counts = unperm_counts[idx]
        these_bins = bins[idx]

        for i, level in enumerate(levels):
            ceil = 1.0 if i == len(levels) - 1 else levels[i + 1] 
            idxs = np.nonzero(
                np.bitwise_and(
                    raw_conf[idx] >= level,
                    raw_conf[idx] <  ceil))

            if len(these_bins[idxs]) > 0:
                conf_to_stat[idx + (i,)]  = these_bins[idxs][0]
                conf_to_count[idx + (i,)] = counts[idxs][0]

    return (conf_to_stat, conf_to_count)

def pick_alphas(conf_to_count, axis=0):
    """Find the tests that maximize the counts.

    Given an ndarray that gives counts for different statistical
    tests, return the indexes of the tests that maximize the counts.

    """
    return np.argmax(conf_to_count, axis=0)


def find_default_alpha(job):
    """
    Return a default value for alpha, using the given data table and
    condition layout.

    """
    baseline_cols = job.conditions[0]
    baseline_data = job.table[:,baseline_cols]

    alphas = np.zeros(len(job.conditions))

    for (c, cols) in enumerate(job.conditions):
        if c == 0: 
            continue

        values = compute_s(job.table[:,cols], baseline_data, axis=1)
        mean = np.mean(values)
        residuals = values[values < mean] - mean
        sd = np.sqrt(sum(residuals ** 2) / (len(residuals) - 1))
        alphas[c] = mean * 2 / np.sqrt(len(cols) + len(baseline_cols))

    return alphas



def all_subsets(n, k):
    """
    Return an (m x n) array where n is the size of the set, and m is
    the number of subsets of size k from a set of size n. 

    Each row is an array of booleans, with k values set to True. For example:

    >>> all_subsets(3, 2)
    array([[ True,  True, False],
           [ True, False,  True],
           [False,  True,  True]], dtype=bool)

    """
    indexes = np.arange(n)
    combinations = list(itertools.combinations(indexes, k))
    result = np.zeros((len(combinations), n), dtype=bool)
    for i, subset in enumerate(combinations):
        result[i, subset] = True
    
    return result


def init_perms(conditions):

    perms = [None]

    baseline_len = len(conditions[0])

    for c in range(1, len(conditions)):
        this_len = len(conditions[c])
        n = baseline_len + this_len
        k = min(baseline_len, this_len)
        perms.append(all_subsets(n, k))

    return perms

def accumulate_bins(bins):
    return np.cumsum(bins[::-1])[::-1]


def get_perm_counts(job, unperm_stats, tests, edges):

    (M, N) = np.shape(unperm_stats)

    all_perms = init_perms(job.conditions)

    shape = list(np.shape(edges))
    shape[1] -= 1
    res = np.zeros(shape)

    n  = len(job.conditions)
    n0 = len(job.conditions[0])

    plt.cla()

    for c in range(1, n):
        logging.info('    Working on condition %d of %d' % (c, n - 1))

        perms = all_perms[c]
        r  = len(perms)
        nc = len(job.conditions[c])
        
        # This is the list of all indexes into table for
        # replicates of condition 0 and condition c.
        master_indexes = np.concatenate(
            (job.conditions[0],
             job.conditions[c]))
        
        for perm_num, perm in enumerate(perms):
            v1 = job.table[:, master_indexes[perm]]
            v2 = job.table[:, master_indexes[~perm]]
            stats = tests[c].compute((v2, v1))
            res[c] += cumulative_hist(stats, edges[c])

        res[c] = res[c] / float(len(all_perms[c]))

    return res


def cumulative_hist(values, bins):
    (hist, ignore) = np.histogram(values, bins)
    return accumulate_bins(hist)

def get_unperm_counts(unperm_stats, edges):

    """Count the number of features with statistic values between the
    given bin edges. unperm_stats is an M x N array where,
    unperm_stats[m, n] gives the statistic value for feature m in
    condition n. Edges is an array of monotonically increasing numbers,
    of length B. Returns a B x N array, where result[b, n] gives the
    number of features that have the statistic between edges[b - 1] and
    edges[b] for condition n.

    TODO: We should be able to make the dimensionality of this
    function flexible.
    """
    logging.debug("Shape of stats is " + str(np.shape(unperm_stats)))
    logging.debug("Shape of edges is " + str(np.shape(edges)))
    (M, N) = np.shape(unperm_stats)
    shape = list(np.shape(edges))
    shape[1] -= 1
    res = np.zeros(shape, int)

    for c in range(1, M):
        res[c] = cumulative_hist(unperm_stats[c], edges[c])

    return res

def uniform_bins(num_bins, stats):

    base_shape = np.shape(stats)[:-1]
    bins = np.zeros(base_shape + (num_bins + 1,))
    for idx in np.ndindex(base_shape):
        maxval = np.max(stats[idx])
        edges = np.concatenate((np.linspace(0, maxval, num_bins), [np.inf]))
        edges[0] = - np.inf
        bins[idx] = edges

    return bins


def custom_bins(unperm_stats):
    """Get an array of bin edges based on the actual computed
    statistic values. unperm_stats is an M x N array, where
    unperm_stats[m, n] gives the statistic value for feature m in
    condition n. Returns an M + 2 x N array, where bins[m, n] and
    bins[m + 1, n] define a bin in which to count features for
    condition n. There is a bin edge for negative and positive
    infinity, and one for each statistic value.

    TODO: We should be able to make the dimensionality of this
    function flexible.
    """

    (S, m, n) = np.shape(unperm_stats)
    bins = np.zeros((S, m + 2, n))
    for i in range(S):
        for c in range(1, n):
            bins[i, 0,     c]  = -np.inf
            bins[i, 1 : m + 1, c] = sorted(unperm_stats[i, :, c])
            bins[i, m + 1, c] = np.inf
    return bins

def raw_confidence_scores(unperm_counts, perm_counts, bins, N):

    logging.debug("Shape of counts is " + str(np.shape(unperm_counts)) + ", bins is " + str(np.shape(bins)))

    shape = np.shape(unperm_counts)
    res = np.zeros(shape)
    for idx in np.ndindex(shape):
        if bins[idx] < 0:
            continue

        R = float(unperm_counts[idx])
        if R > 0:
            V = R - adjust_num_diff(perm_counts[idx], R, N)
            res[idx] = V / R
    return res
      
          
def do_confidences_by_cutoff(job, default_alphas, num_bins):

    # Some values used as sizes of axes
    N = len(job.table)
    C = len(job.conditions)
    S = len(stats.Tstat.TUNING_PARAM_RANGE_VALUES)

    base_shape = (S, C)

    tests = np.zeros(base_shape, dtype=object)

    # Unperm stats gives the value of the statistic for each test, for
    # each feature, in each condition.
    unperm_stats = np.zeros(base_shape + (N,))

    for (i, j) in np.ndindex(base_shape):
        tests[i, j] = stats.Tstat(
            stats.Tstat.TUNING_PARAM_RANGE_VALUES[i] * default_alphas[j])

    print "Getting stats for unpermuted data"
    for i in range(len(stats.Tstat.TUNING_PARAM_RANGE_VALUES)):
        unperm_stats[i] = unpermuted_stats(job, tests[i])

    up = compute_directional_results(job, tests, unperm_stats)
    down = compute_directional_results(job, tests, -unperm_stats)

    return IntermediateResults(
        default_alphas, unperm_stats, job.levels, up, down)        
    

def compute_directional_results(job, tests, unperm_stats):

#    edges = custom_bins(unperm_stats)
    edges = uniform_bins(1001, unperm_stats)

    base_shape = np.shape(unperm_stats)[:-1]
    num_bins = np.shape(edges)[-1]
    logging.debug("Shape of edges is " + str(np.shape(edges)))
    perm_counts   = np.zeros(base_shape + (num_bins - 1,))
    unperm_counts = np.zeros_like(perm_counts)

    print "Getting counts for permuted data"
    for i, test_row in enumerate(tests):

        perm_counts[i] = get_perm_counts(job, unperm_stats[i], test_row, edges[i])
        unperm_counts[i] = get_unperm_counts(unperm_stats[i], edges[i])

    print "Getting raw confidence scores in {num_edges} edges".format(
        num_edges=np.shape(edges)[-1])
    raw_conf = raw_confidence_scores(
        unperm_counts, perm_counts, edges, len(job.table))

    print "Summarizing confidence scores for {num_levels} levels".format(
        num_levels=len(job.levels))
    (conf_to_stat, conf_to_count) = summarize_confidence(
        job.levels, unperm_counts, raw_conf, edges)

    best_params = pick_alphas(conf_to_count)

    return DirectionalResults(
        unperm_counts,
        raw_conf, 
        conf_to_stat,
        conf_to_count,
        best_params)
    

def plot_cumulative(data):
    (m, n) = np.shape(data)
    plt.clf()
    for c in range(1, n):
        col = sorted(data[:, c])
        plt.plot(col, np.arange(m))
    plt.savefig("cumulative")


def ensure_increases(a):
    """Given an array, return a copy of it that is monotonically
    increasing."""

    for i in range(len(a) - 1):
        a[i+1] = max(a[i], a[i+1])


def adjust_num_diff(V0, R, num_ids):
    V = np.zeros(6)
    V[0] = V0
    for i in range(1, 6):
        V[i] = V[0] - V[0] / num_ids * (R - V[i - 1])
    return V[5];

def unpermuted_stats(job, statfns):

    stats = np.zeros((len(job.conditions),
                      len(job.table)))
                      
    for c in range(1, len(job.conditions)):
        v1 = job.table[:, job.conditions[0]]
        v2 = job.table[:, job.conditions[c]]
        stats[c] = statfns[c].compute((v2, v1))

    return stats

