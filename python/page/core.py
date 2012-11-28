#!/usr/bin/env python



"""
m is the number of features
n is the number of conditions
n_i is the number of replicates in the ith condition

h is the number of bins
r is the number of permutations
s is the number of tuning param range values

Plot:

* Histogram of feature count by statistic, one for each condition /
  alpha.

* Cumulative num features by statistic. One chart for each
  condition, with different alphas.

* Num features for each confidence level, one for each condition, with
  different alphas. Overlay "best" alpha at each confidence level.

"""
import matplotlib
import matplotlib.pyplot as plt
import re
import itertools 
import logging
import numpy as np
import logging

from stats import Tstat
from report import Report

class Job(object):

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

########################################################################
###
### Constants
###

__version__ = '6.0.0'

TUNING_PARAM_RANGE_VALUES = np.array([
    0.0001,
    0.01,
    0.1,
    0.3,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    10,
    ])

TUNING_PARAM_RANGE_VALUES = np.array([0.5])

TUNING_PARAM_RANGE_VALUES = np.array([0.0001, 0.5, 10])

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


def get_permuted_means(job, mins, maxes, tests, num_bins=1000):
    all_perms = init_perms(job.conditions)

    h  = num_bins
    n  = len(job.conditions)
    n0 = len(job.conditions[0])

    # conditions x bins, typically 10 2 x 1000 = 2000. Not too big.
    mean_perm_u = np.zeros((n, h + 1))
    mean_perm_d = np.zeros((n, h + 1))

    for c in range(1, n):
        print '    Working on condition %d of %d' % (c, n - 1)
        perms = all_perms[c]
        r  = len(perms)
        nc = len(job.conditions[c])
        
        # This is the list of all indexes into table for
        # replicates of condition 0 and condition c.
        master_indexes = np.zeros((n0 + nc), dtype=int)
        master_indexes[:n0] = job.conditions[0]
        master_indexes[n0:] = job.conditions[c]
        
        # Histogram is (permutations x bins)
        hist_u = np.zeros((r, h + 1), int)
        hist_d = np.zeros((r, h + 1), int)

        for perm_num, perm in enumerate(perms):

            v1 = job.table[:, master_indexes[perm]]
            v2 = job.table[:, master_indexes[~perm]]

            stats = tests[c].compute((v2, v1))
            (hist_u[perm_num], hist_d[perm_num]) = assign_bins(stats, h, mins[c], maxes[c])

        # Bin 0 is for features that were downregulated (-inf, 0) Bins
        # 1 through 999 are for features that were upregulated Bin
        # 1000 is for any features that were upregulated above the max
        # from the unmpermuted data (max, inf)

        mean_perm_u[c] = np.mean(hist_u, axis=0)
        mean_perm_d[c] = np.mean(hist_d, axis=0)
        

    for c in range(1, n):
        y = mean_perm_u[c]
        x = range(len(y))
        plt.plot(x[500:], y[500:])

    plt.savefig('mean_perm_u')
    plt.xlabel('bin')
    plt.ylabel('up-regulated features')
    
    return (mean_perm_u, mean_perm_d)


def get_perm_counts(job, unperm_stats, tests, bins):
    """

    Returns two m x n arrays
    """

    (m, n) = np.shape(unperm_stats)
    all_perms = init_perms(job.conditions)
    res = np.zeros((m + 1, n))
    n  = len(job.conditions)
    n0 = len(job.conditions[0])

    plt.cla()

    for c in range(1, n):
        print '    Working on condition %d of %d' % (c, n - 1)

        perms = all_perms[c]
        r  = len(perms)
        nc = len(job.conditions[c])
        
        # This is the list of all indexes into table for
        # replicates of condition 0 and condition c.
        master_indexes = np.zeros((n0 + nc), dtype=int)
        master_indexes[:n0] = job.conditions[0]
        master_indexes[n0:] = job.conditions[c]
        
        for perm_num, perm in enumerate(perms):

            v1 = job.table[:, master_indexes[perm]]
            v2 = job.table[:, master_indexes[~perm]]

            stats = tests[c].compute((v2, v1))

            (hist, edges) = np.histogram(stats, bins[:, c])
            res[:, c] += accumulate_bins(hist)

        res[:, c] = res[:, c] / float(len(all_perms[c]))

        plt.plot(bins[1:, c], res[:, c])
    plt.savefig('new_unperm_counts')

    return (res, bins)


def get_unperm_counts(unperm_stats, bins):

    (m, n) = np.shape(unperm_stats)
    res = np.zeros((m + 1, n))

    for c in range(1, n):

        (hist, edges) = np.histogram(unperm_stats[:, c], bins=bins[:, c])
        res[:, c] = accumulate_bins(hist)

    return (res, bins)


def custom_bins(unperm_stats):

    (m, n) = np.shape(unperm_stats)
    bins = np.zeros((m + 2, n))
    for c in range(1, n):
        bins[0,     c]  = -np.inf
        bins[1 : m + 1, c] = sorted(unperm_stats[:, c])
        bins[m + 1, c] = np.inf
    return bins

def raw_confidence_scores(unperm_counts, perm_counts, bins):
    print "Shape of counts is " + str(np.shape(unperm_counts)) + ", bins is " + str(np.shape(bins))
    shape = np.shape(unperm_counts)
    res = np.zeros(shape)
    for idx in np.ndindex(shape):
        (i, j, c) = idx
        if bins[idx] < 0:
            continue

        R = float(unperm_counts[idx])
        if R > 0:
            V = R - adjust_num_diff(perm_counts[idx], R, shape[1])
            res[idx] = V / R
    return res
                

def fdr(unperm_stats, unperm_counts, perm_counts, bins):

    (m, n) = np.shape(unperm_stats)

    # Columns:

    """
    stat - Value of the statistic.

    ptn - The proportion of the statistic to the maximum value seen in
          the unpermuted data.

    perm - Number of features for which the statistic is greater than
           this statistic, in the permuted data.

    unperm - Same, for the unpermuted data.

    conf - The confidence score.

    """

    plt.cla()

    for c in range(1, n):

        disc = []
        for i in range(len(bins[:, c]) - 1):
            disc.append((bins[i, c], 0.0, perm_counts[i, c], unperm_counts[i, c], 0.0))

        disc = np.array(disc, dtype=[('stat',   float),
                                     ('ptn',    float),
                                     ('perm',   int),
                                     ('unperm', int),
                                     ('conf',   float)])

        for row in disc:
            R = float(row['unperm'])
            if R > 0:
                V = R - adjust_num_diff(row['perm'], R, m)
                row['conf'] = V / R

        disc['ptn'] = disc['stat'] / np.max(unperm_stats[:, c])

        np.savetxt('table_' + str(c), disc, ['%.10f', '%.4f', '%d', '%d', '%.4f'])

        plt.plot(disc['conf'],
                 disc['unperm'])

    plt.ylim([0, 100])
    plt.xlim([0.5, 1])
    plt.xlabel("Confidence score")
    plt.ylabel("Num features up-regulated")
    plt.savefig("features_by_conf")
                 

def make_confidence_bins(unperm, perm, num_features):
    conf_bins = np.zeros(np.shape(unperm))
    for idx in np.ndindex(np.shape(unperm)):
        R = unperm[idx]
        if R > 0:
            V = R - adjust_num_diff(perm[idx], R, num_features)
            conf_bins[idx] = V / R

    for idx in np.ndindex(np.shape(conf_bins)[0:1]):
        ensure_increases(conf_bins[idx])

    return conf_bins

def do_confidences_by_cutoff(job, default_alphas, num_bins):

    table      = job.table
    conditions = job.conditions

    print "Getting stats for unpermuted data"

    gene_conf_u = np.zeros((len(TUNING_PARAM_RANGE_VALUES),
                            len(table),
                            len(conditions)))
    gene_conf_d = np.zeros(np.shape(gene_conf_u))

    conf_bins_u = np.zeros((len(TUNING_PARAM_RANGE_VALUES),
                            len(conditions),
                            num_bins + 1))
    conf_bins_d = np.zeros(np.shape(conf_bins_u))

    c0 = table[:,conditions[0]]

    # Unperm stats gives the value of the statistic for each test, for
    # each feature, in each condition.
    unperm_stats = np.zeros((len(TUNING_PARAM_RANGE_VALUES),
                             len(table),
                             len(conditions)))

    perm = np.zeros((len(TUNING_PARAM_RANGE_VALUES),
                     len(table) + 1,
                     len(conditions)))

    unperm_counts = np.zeros((len(TUNING_PARAM_RANGE_VALUES),
                              len(table) + 1,
                              len(conditions)))

    bins = np.zeros((len(TUNING_PARAM_RANGE_VALUES),
                     len(table) + 2,
                     len(conditions)))

    for i, alpha_mul in enumerate(TUNING_PARAM_RANGE_VALUES):
        print "Trying alpha multiplier " + str(alpha_mul)

        tests = [Tstat(a * alpha_mul) for a in default_alphas]

        mins  = np.zeros(len(conditions))
        maxes = np.zeros(len(conditions))

        print "  Finding mins and maxes"
        # Find the minimum and maximum value (across all features) of
        # the statistic for each condition
        for c in range(1, len(conditions)):
            data = table[:, conditions[c]]
            vals = tests[c].compute((data, c0))
            mins[c]  = np.min(vals)
            maxes[c] = np.max(vals)

        print "  Getting means of statistic on permuted columns"
        # Try some permutations of the columns, and take the average
        # statistic for up- and down- regulated features.
        (mean_perm_u, mean_perm_d) = get_permuted_means(
            job, mins, maxes, tests, 1000)

        print "  Getting unpermuted statistics"
        # Now get the statistic for the unpermuted data
        (num_unperm_u, num_unperm_d, unperm_stats[i]) = unpermuted_stats(
            job, mins, maxes, tests, 1000)

        print "  Making confidence bins"
        conf_bins_u[i] = make_confidence_bins(
            num_unperm_u, mean_perm_u, len(table))
        conf_bins_d[i] = make_confidence_bins(
            num_unperm_d, mean_perm_d, len(table))

        plt.cla()
        for c in range(1, 4):
            plt.plot(np.arange(len(mean_perm_u[i])),
                     mean_perm_u[c])
        plt.xlabel("Percent of max statistic")
        plt.ylabel("Features above statistic")
        plt.xlim([0, 1000])
        plt.ylim([0, 1000])
        plt.savefig("old_perm" + str(c))

        print "Computing confidence scores"
        (gene_conf_u[i], gene_conf_d[i]) = get_gene_confidences(
            unperm_stats[i], mins, maxes, conf_bins_u[i], conf_bins_d[i])

        print "New stuff"
        bins[i] = custom_bins(unperm_stats[i])
        (perm[i],     perm_bins) = get_perm_counts(job, unperm_stats[i], tests, bins[i])
        
        (unperm_counts[i], unperm_bins) = get_unperm_counts(unperm_stats[i], bins[i])

        fdr(unperm_stats[i], unperm_counts[i], perm[i], bins[i])


    raw_conf = raw_confidence_scores(unperm_counts, perm, bins)

    report = Report('page_output', unperm_stats, unperm_counts, raw_conf)
    report.make_report()

    np.save("alpha", default_alphas)
    np.save("gene_conf_u", gene_conf_u)
    np.save("gene_conf_d", gene_conf_d)
    
    print "Counting up- and down-regulated features in each level"
    logging.info("Shape of conf up is " + str(np.shape(gene_conf_u)))
    levels = np.linspace(0.5, 0.95, 10)

    breakdown = breakdown_tables(levels, gene_conf_u, gene_conf_d)
    logging.info("Levels are " + str(levels))
    return (conf_bins_u, conf_bins_d, breakdown)


def plot_cumulative(data):
    (m, n) = np.shape(data)
    plt.clf()
    for c in range(1, n):
        col = sorted(data[:, c])
        plt.plot(col, np.arange(m))
    plt.savefig("cumulative")



def proportions(numer, denom):
    (m, n) = np.shape(numer)

    prop_u = np.zeros((m, n))
    prop_d = np.zeros((m, n))

    for c in range(1, n):
        prop_u[:, c] = numer[:, c] / np.max(denom[:, c])
        prop_d[:, c] = numer[:, c] / np.min(denom[:, c])

    prop_u[prop_u < 0] = 0.0
    prop_d[prop_d < 0] = 0.0
    return (prop_u, prop_d)

def perm_propertions(perm_stats, unperm_stats):
    (m, n) = np.shape(unperm_stats)

    prop_u = np.zeros((m, n))
    prop_d = np.zeros((m, n))

    for c in range(1, n):
        prop_u[:, c] = col / np.max(col)
        prop_d[:, c] = col / np.min(col)

    prop_u[prop_u < 0] = 0.0
    prop_d[prop_d < 0] = 0.0
    return (prop_u, prop_d)

def count_stats(unperm_stats, perm_stats):

    (m, n) = np.shape(unperm_stats)

    for c in range(1, n):
        perm   = perm_stats[:, c]
        unperm = unperm_stats[:, c]
        idx    = range(len(perm))

        
        
#        plt.clf()
#        
#        plt.savefig("stat_diff_" + str(c))

def ensure_increases(a):
    """Given an array, return a copy of it that is monotonically
    increasing."""

    for i in range(len(a) - 1):
        a[i+1] = max(a[i], a[i+1])


def breakdown_tables(levels, gene_conf_u, gene_conf_d):
    """u_by_conf gives the number of up-regulated features for each
    combination of alpha, condition, and confidence
    level. d_by_conf gives the same for dow-regulated features."""

    u_by_conf = get_count_by_conf_level(gene_conf_u, levels)
    d_by_conf = get_count_by_conf_level(gene_conf_d, levels)

    (num_range_values, n, num_levels) = np.shape(u_by_conf)

    # For each condition and confidence level, find the values of
    # alpha that give the maximum number of up- and down- regulated
    # features.
    max_up_params   = np.argmax(u_by_conf, axis=0)
    max_down_params = np.argmax(d_by_conf, axis=0)
 
    breakdown = np.zeros((n, len(levels), 3))

    for c in range(1, n):
            
        breakdown[c, :, 0] = levels
        for i in range(len(levels)):
            breakdown[c, i, 1] = u_by_conf[max_up_params[c, i], c, i]
            breakdown[c, i, 2] = d_by_conf[max_down_params[c, i], c, i]

    return breakdown


def get_count_by_conf_level(gene_conf, ranges):

    (num_range_values, num_genes, num_conditions) = np.shape(gene_conf)
    shape = (num_range_values, num_conditions, len(ranges))

    u_by_conf   = np.zeros(shape)
    
    for i in range(num_range_values):
        for j in range(num_conditions):
            up_conf   = gene_conf  [i, :, j]
            for (k, level) in enumerate(ranges):
                u_by_conf  [i, j, k] = len(up_conf  [up_conf   > level])

    return u_by_conf


def get_gene_confidences(unperm_stats, mins, maxes, conf_bins_u, conf_bins_d):
    """Returns a pair of 3D arrays: gene_conf_u and
    gene_conf_d. gene_conf_u[i, j, k] indicates the confidence
    with which gene j is upregulated in condition k using the ith
    alpha multiplier. gene_conf_d does the same thing for
    down-regulation."""

    num_bins    = np.shape(conf_bins_u)[-1] - 1
    gene_conf_u = np.zeros(np.shape(unperm_stats))
    gene_conf_d = np.zeros(np.shape(unperm_stats))

    for idx in np.ndindex(np.shape(unperm_stats)):
        (j, c) = idx
        if c == 0:
            continue
        if unperm_stats[idx] >= 0:			
            binnum = int(num_bins * unperm_stats[idx] / maxes[c])
            gene_conf_u[idx] = conf_bins_u[c, binnum]
        else:
            binnum = int(num_bins * unperm_stats[idx] / mins[c])
            gene_conf_d[idx] = conf_bins_d[c, binnum]

    return (gene_conf_u, gene_conf_d)


def adjust_num_diff(V0, R, num_ids):
    V = np.zeros(6)
    V[0] = V0
    for i in range(1, 6):
        V[i] = V[0] - V[0] / num_ids * (R - V[i - 1])
    return V[5];


def assign_bins(vals, num_bins, minval, maxval):
    """
    Computes two np.histograms for the given values.
    """
    u_bins = get_bins(num_bins + 1,  maxval)
    d_bins = get_bins(num_bins + 1, -minval)

    (u_hist, u_edges) = np.histogram( vals, u_bins)
    (d_hist, d_edges) = np.histogram(-vals, d_bins)
    u_hist[0] += len(vals[vals < 0.0])
    d_hist[0] += len(vals[vals > 0.0])

    return (accumulate_bins(u_hist),
            accumulate_bins(d_hist))

def unpermuted_stats(job, mins, maxes, statfns, num_bins):

    u = np.zeros((len(job.conditions), num_bins + 1), dtype=int)
    d = np.zeros(np.shape(u), int)

    stats = np.zeros((len(job.table), len(job.conditions)))

    for c in range(1, len(job.conditions)):
        v1 = job.table[:, job.conditions[0]]
        v2 = job.table[:, job.conditions[c]]
        stats[:, c] = statfns[c].compute((v2, v1))

        (u[c], d[c]) = assign_bins(stats[:, c], num_bins, mins[c], maxes[c])

    return (u, d, stats)

def get_bins(n, maxval):

    # Bin 0 in the "up" histogram is for features that were down-regulated
    bins = []
    bins.extend(np.linspace(0, maxval, n))

    # Bin "numbin" in the "up" histogram is for features that were
    # above the max observed in the unpermuted data
    bins.append(np.inf)
    return bins
