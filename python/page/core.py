#!/usr/bin/env python



"""

PaGE uses a lot of multi-dimensional arrays. When possible, the
dimensions should be ordered as follows:

  + Direction (up or down)
  + Test (e.g. different values of alpha)
  + Confidence level
  + Class (or condition)
  + Feature number
  + Bin (for discretizing the distibution of statistic)

For t-test, shape is (test, condition)
For f-test, shape is (test,)

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

DirectionalResults = collections.namedtuple(
    'DirectionalResults',
    'edges unperm_counts raw_conf conf_to_stat conf_to_count best_params')

class Results:
    """Holds the intermediate results of a job.

    alphas - TODO
    
    stats - a T x C x N array, where T is the number of tests, C is
            the number of conditions, and N is the number of features.

    conf_levels - an array of floats of length L, representing the
                  (lower) edges of the confidence levels.

    """
    def __init__(self, alphas, stats, conf_levels, best_params,
                 conf_to_stat, conf_to_count, raw_conf, edges):
        self.alphas = alphas
        self.stats  = stats
        self.conf_levels = conf_levels
        self.best_params = best_params
        self.conf_to_stat = conf_to_stat
        self.conf_to_count = conf_to_count
        self.raw_conf = raw_conf
        self.edges = edges

    def save(self, output_dir):
        cwd = os.getcwd()
        try:
            os.chdir(output_dir)
            np.save('alphas', self.alphas)
            np.save('stats', self.stats)
            np.save('conf_levels', self.conf_levels)
            np.save('best_params', self.best_params)
            np.save('conf_to_stat', self.conf_to_stat)
            np.save('conf_to_count', self.conf_to_count)
            np.save('raw_conf', self.raw_conf)
            np.save('edges', self.edges)

        finally:
            os.chdir(cwd)

    @classmethod
    def load(cls, dir_):
        """Load saved results from the given directory."""

        cwd = os.getcwd()
        try:
            os.chdir(dir_)
            alphas        = np.load('alphas.npy')
            stats         = np.load('stats.npy')
            conf_levels   = np.load('conf_levels.npy')
            best_params   = np.load('best_params.npy')
            conf_to_stat  = np.load('conf_to_stat.npy')
            conf_to_count = np.load('conf_to_count.npy')
            raw_conf      = np.load('raw_conf.npy')
            edges         = np.load('edges.npy')
            return Results(alphas, stats, conf_levels, best_params,
                           conf_to_stat, conf_to_count, raw_conf, edges)
        finally:
            os.chdir(cwd)

    @property 
    def num_directions(self):
        """Number of directions, typically 1 or 2."""
        return 2

    @property
    def num_levels(self):
        """Number of confidence levels for the job."""
        return len(self.conf_levels)
    
    @property
    def num_tests(self):
        """Number of tests (values of alpha) for the job."""
        return np.shape(self.stats)[0]

    @property
    def num_classes(self):
        """Number of classes (conditions)."""
        return np.shape(self.stats)[1]

    @property
    def num_features(self):
        """Number of features (e.g. genes)."""
        return np.shape(self.stats)[2]

    @property
    def cutoffs_by_level(self):
        """A direction x level x class array.

        cutoffs_by_level[d, l, c] gives the value of statistic to use
        as the cutoff for determining whether a feature is regulated
        in direction d (0 is up, 1 is down) in class c with confidence
        corresponding to confidence level l.

        """

        res = np.zeros((self.num_directions, self.num_levels, self.num_classes))

        for idx in np.ndindex(np.shape(res)):
            (d, l, c) = idx
            cutoff = self.conf_to_stat[d, self.best_params[idx], c, l]
            res[idx] = cutoff
        return res

    @property
    def feature_to_conf_by_conf(self):

        """A direction x level x class x feature array.

        feature_to_conf_by_conf[d, l, c, i] gives the confidence that
        feature i is regulated in direction d (0 is up, 1 is down) in
        class c for confidence level l.

        """
        res = np.zeros((2, self.num_levels, self.num_classes, self.num_features))
        feature_to_conf = self.feature_to_conf
        for idx in np.ndindex(np.shape(res)[:-1]):
            src_idx = (idx[0], self.best_params[idx], idx[2])
            res[idx] = feature_to_conf[src_idx]

        return res


    @property
    def feature_to_conf(self):
        """A test x class x feature array."""

        res = np.zeros((self.num_directions, self.num_tests, 
                        self.num_classes, self.num_features))

        for d, i, j in np.ndindex(np.shape(res)[:-1]):
            table = [(k, v) for (k, v) in enumerate(self.stats[i, j])]
            table = np.array(table, dtype=[
                    ('feature', int),
                    ('stat', float)])
            table = np.sort(table, order='stat')
            
            edgenum = 0

            for (k, stat) in table:
                if stat < self.edges[d, i, j, edgenum]:
                    raise "Edges are not increasing"

                while stat >= self.edges[d, i, j, edgenum + 1]:
                    edgenum += 1
                    
                res[d, i, j, k] = self.raw_conf[d, i, j, edgenum]
        return res

    @property
    def best_counts(self):

        res = np.zeros((self.num_directions, self.num_levels, self.num_classes))
        for idx in np.ndindex(np.shape(res)):
            (d, l, c) = idx
            test_idx = self.best_params[idx]
            res[idx] = self.conf_to_count[(d, test_idx, idx[2], idx[1])]
        return res

    @property
    def best_stats_by_level(self):
        """The statistics (with optimal alpha) at each level.

        A direction x level x class x feature array. For example,
        best_stats_by_level[UP, 1, 2, 3] gives the statistic for
        feature 3 using the 'best' value of alpha for up-regulation at
        confidence level 1 in class 2.

        """
        res = np.zeros((2,
                        self.num_levels, 
                        self.num_classes, 
                        self.num_features))

        for (d, i, c) in np.ndindex(np.shape(res)[:-1]):
            res[d, i, c] = self.stats[self.best_params[d, i, c], c]
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

    def new_table(self):
        conds = self.conditions
        table = self.table
        res = np.zeros((len(conds),
                        len(table),
                        len(conds[0])))

        for i, c in enumerate(conds):
            res[i] = table[:, c]
        return np.swapaxes(res, 1, 2)

########################################################################
###
### Constants
###

__version__ = '6.0.0'


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
    Return the lowest statistic at which the confidence level is
    greater or equal to conf.

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
            else:
                conf_to_stat[idx + (i,)] = np.inf

    return (conf_to_stat, conf_to_count)

def pick_alphas(conf_to_count, axis=0):
    """Find the tests that maximize the counts.

    Given an ndarray that gives counts for different statistical
    tests, return the indexes of the tests that maximize the counts.

    """
    return np.swapaxes(np.argmax(conf_to_count, axis=0), 0, 1)


def find_default_alpha(job):
    """
    Return a default value for alpha, using the given data table and
    condition layout.

    """

    table = job.new_table()
    alphas = np.zeros(len(table))
    (num_classes, samples_per_class, num_features) = np.shape(table)

    for c in range(1, num_classes):
        values = compute_s(table[c], table[0])
        mean = np.mean(values)
        residuals = values[values < mean] - mean
        sd = np.sqrt(sum(residuals ** 2) / (len(residuals) - 1))
        alphas[c] = mean * 2 / np.sqrt(samples_per_class * 2)

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

        table = job.new_table()
        table = np.vstack((table[0], table[c]))

        for perm_num, perm in enumerate(all_perms[c]):
            data = concat_directions(table[~perm], table[perm])
            stats = tests[c].compute(data)
            res[c] += cumulative_hist(stats, edges[c])

        res[c] = res[c] / float(len(all_perms[c]))

    return res


def cumulative_hist(values, bins):
    (hist, ignore) = np.histogram(values, bins)
    return np.cumsum(hist[::-1])[::-1]


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
    """Calculate the confidence scores.

    Both unperm_counts and perm_counts are (test x class x bins)
    arrays, and bins is a (test x class x bins + 1) array. N is the
    number of features.

    Returns a 3-d array where res[test, class, bin] is the confidence
    that the features that have the statistic between bins[bin] and
    bins[bin + 1] are differentially expressed in class, using the
    given test.
    """
    
    logging.info("Shape of counts is " + str(np.shape(unperm_counts)) + 
                 ", bins is " + str(np.shape(bins)))

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


def concat_directions(up, down):
    return np.concatenate((up, down)).reshape((2,) + np.shape(up))

          
def do_confidences_by_cutoff(job, default_alphas, num_bins):

    # Some values used as sizes of axes
    table = job.new_table()
    (num_classes, num_samples, num_features) = np.shape(table)
    num_tests = len(stats.Tstat.TUNING_PARAM_RANGE_VALUES)

    base_shape = (num_tests, num_classes)

    alphas = np.zeros(base_shape)
    tests = np.zeros(base_shape, dtype=object)

    # Unperm stats gives the value of the statistic for each test, for
    # each feature, in each condition.
    unperm_stats = np.zeros(base_shape + (num_features,))

    for (i, j) in np.ndindex(base_shape):
        alphas[i, j] = stats.Tstat.TUNING_PARAM_RANGE_VALUES[i] * default_alphas[j]
        tests[i, j] = stats.Tstat(alphas[i, j])


    print "Getting stats for unpermuted data"
    for i, test_row in enumerate(tests):
        unperm_stats[i] = unpermuted_stats(job, test_row)

    up   = compute_directional_results(job, tests,  unperm_stats)
    down = compute_directional_results(job, tests, -unperm_stats)
    
    best_params   = concat_directions(up.best_params, down.best_params)
    conf_to_stat  = concat_directions(up.conf_to_stat, down.conf_to_stat)
    conf_to_count = concat_directions(up.conf_to_count, down.conf_to_count)
    raw_conf      = concat_directions(up.raw_conf, down.raw_conf)
    edges         = concat_directions(up.edges, down.edges)

    return Results(
        alphas, unperm_stats, job.levels, best_params, conf_to_stat,
        conf_to_count, raw_conf, edges)
    

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

    new_len = np.shape(job.new_table())[2]
    
    raw_conf = raw_confidence_scores(
        unperm_counts, perm_counts, edges, new_len)

    print "Summarizing confidence scores for {num_levels} levels".format(
        num_levels=len(job.levels))
    (conf_to_stat, conf_to_count) = summarize_confidence(
        job.levels, unperm_counts, raw_conf, edges)

    best_params = pick_alphas(conf_to_count)
    print "Best params are " + str(best_params)
    return DirectionalResults(
        edges,
        unperm_counts,
        raw_conf, 
        conf_to_stat,
        conf_to_count,
        best_params)
    

def adjust_num_diff(V0, R, num_ids):
    V = np.zeros(6)
    V[0] = V0
    for i in range(1, 6):
        V[i] = V[0] - V[0] / num_ids * (R - V[i - 1])
    return V[5];

def unpermuted_stats(job, statfns):

    table = job.new_table()
    (num_conditions, samples_per_cond, num_features) = np.shape(table)
    stats = np.zeros((num_conditions, num_features))
    for c in range(1, num_conditions):
        data = table[([c, 0],)]
        stats[c] = statfns[c].compute(data)

    return stats

