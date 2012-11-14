#!/usr/bin/env python

"""
m is the number of features
n is the number of conditions
n_i is the number of replicates in the ith condition

h is the number of bins
r is the number of permutations
s is the number of tuning param range values

"""

import re
import itertools 
import logging

import numpy as np
import logging

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


########################################################################
###
### Functions
###

def load_input(fh):

    """Read input from the given filehandle and return the data, the
    feature ids, and the layout of the conditions and replicates.

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

    return (table, ids)

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

def find_default_alpha(table, conditions):
    """
    Return a default value for alpha, using the given data table and
    condition layout.
    """

    baseline_cols = conditions[0]
    baseline_data = table[:,baseline_cols]

    alphas = np.zeros(len(conditions))

    for (c, cols) in enumerate(conditions):
        if c == 0: 
            continue

        values = compute_s(table[:,cols], baseline_data, axis=1)
        mean = np.mean(values)
        residuals = values[values < mean] - mean
        sd = np.sqrt(sum(residuals ** 2) / (len(residuals) - 1))
        alphas[c] = mean * 2 / np.sqrt(len(cols) + len(baseline_cols))

    return alphas


def tstat(v1, v2, alphas):
    """
    Computes the t-statistic across two vertical slices of the data
    table, with different values of alpha.

    v1 is an m x n1 array and v2 is an m x n2 array, where m is the
    number of features, n1 is the number of replicates in the
    condition represented by v1, and n2 is the number of replicates
    for v2. Returns an (m x s) array, where m again is the number of
    features, and s is the length of the tuning param array.
    """

    # n1 and n2 are the length of each row. TODO: When we start using
    # masked values we will need to use the number of unmasked values
    # in each row. Until then, all the lengths are the same.
    s = len(alphas)
    m = len(v1)
    n1 = np.array([len(row) for row in v1])
    n2 = np.array([len(row) for row in v2])

    # Variance for each row of v1 and v2 with one degree of
    # freedom. var1 and var2 will be 1-d arrays, one variance for each
    # feature in the input.
    var1 = np.var(v1, ddof=1, axis=1)
    var2 = np.var(v2, ddof=1, axis=1)

    S = np.sqrt((var1 * (n1-1) + var2 * (n2-1)) /(n1 + n2 - 2))

    # This just makes an s x n array where each column is a copy of
    # alpha, and another s x n array where each row is a copy of foo. We
    # do this so they're the same shape, so we can add them.
    alphas = np.tile(alphas, (m, 1)).transpose()
    S      = np.tile(S, (s, 1))

    numer  = (np.mean(v1, axis=1) - np.mean(v2, axis=1)) * np.sqrt(n1 * n2)
    denom = (alphas + S) * np.sqrt(n1 + n2)

    return numer / denom

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

def min_max_stat(data, conditions, default_alphas):
    """
    Returns a tuple (mins, maxes) where both mins and maxes are (s x
    n) matrices, s being the length of default_alphas, and n being the
    number of conditions.
    """

    m = len(data)
    n = len(conditions)
    s = len(TUNING_PARAM_RANGE_VALUES)

    table = np.zeros((n, s, m))

    for j in range(1, n):
        alphas = default_alphas[j] * TUNING_PARAM_RANGE_VALUES
        table[j,:,:] = tstat(data[:,conditions[j]],
                             data[:,conditions[0]],
                             alphas)

    mins  = np.min(table, axis=2)
    maxes = np.max(table, axis=2)

    return (np.transpose(mins), np.transpose(maxes))

def accumulate_bins(bins):
    return np.cumsum(bins[::-1])[::-1]

def do_confidences_by_cutoff(table, conditions, default_alphas, num_bins):

    all_perms = init_perms(conditions)

    m  = len(table)
    s  = len(TUNING_PARAM_RANGE_VALUES)
    h  = num_bins
    n  = len(conditions)
    n0 = len(conditions[0])

    # tuning params x conditions x bins, typically 10 x 2 x 1000 =
    # 20000. Not too big.
    mean_perm_up   = np.zeros((s, n, h + 1))
    mean_perm_down = np.zeros((s, n, h + 1))

    for c in range(1, n):
        print 'Working on condition %d of %d' % (c, n - 1)
        perms = all_perms[c]
        r  = len(perms)
        nc = len(conditions[c])

        # This is the list of all indexes into table for
        # replicates of condition 0 and condition c.
        master_indexes = np.zeros((n0 + nc), dtype=int)
        master_indexes[:n0] = conditions[0]
        master_indexes[n0:] = conditions[c]

        # Histogram is (permutations x alpha tuning params x bins)
        hist_shape = (r, s, h + 1)
        up   = np.zeros(hist_shape, int)
        down = np.zeros(hist_shape, int)

        (mins, maxes) = min_max_stat(table, conditions, default_alphas)

        # print "  Permuting indexes"
        for perm_num, perm in enumerate(perms):

            v1 = table[:, master_indexes[perm]]
            v2 = table[:, master_indexes[~perm]]
            stats = tstat(v2, v1, default_alphas[c] * TUNING_PARAM_RANGE_VALUES)

            for i in range(s):
                (u_hist, d_hist) = assign_bins(stats[i, :], h, 
                                               mins[i, c], maxes[i, c])
                up  [perm_num, i] = u_hist
                down[perm_num, i] = d_hist

        # Bin 0 is for features that were downregulated (-inf, 0) Bins
        # 1 through 999 are for features that were upregulated Bin
        # 1000 is for any features that were upregulated above the max
        # from the unmpermuted data (max, inf)

        for idx in np.ndindex(len(perms), s):
            up[idx]   = accumulate_bins(up[idx])
            down[idx] = accumulate_bins(down[idx])

#        for perm_num, perm in enumerate(perms):
#            for i in range(s):


        mean_perm_up  [:, c, :] = np.mean(up, axis=0)
        mean_perm_down[:, c, :] = np.mean(down, axis=0)

    print "Getting stats for unpermuted data"
    (num_unperm_up, num_unperm_down, unperm_stats) = dist_unpermuted_stats(table, conditions, mins, maxes, default_alphas)

    for idx in np.ndindex(s, len(conditions)):
        num_unperm_up[idx]   = accumulate_bins(num_unperm_up[idx])
        num_unperm_down[idx] = accumulate_bins(num_unperm_down[idx])

    null_shape = (s, n, h + 1)
    num_null_up   = np.zeros(null_shape)
    num_null_down = np.zeros(null_shape)

    for idx in np.ndindex(null_shape):
        num_null_up[idx] = adjust_num_diff(
            mean_perm_up[idx],
            num_unperm_up[idx],
            m)
        num_null_down[idx] = adjust_num_diff(
            mean_perm_down[idx],
            num_unperm_down[idx],
            m)
                
    conf_bins_up = np.zeros(null_shape)
    conf_bins_down = np.zeros(null_shape)

    for idx in np.ndindex(s, n, h + 1):
        unperm_up = num_unperm_up[idx]
        unperm_down = num_unperm_down[idx]
        if unperm_up > 0:
            conf_bins_up[idx] = (unperm_up - num_null_up[idx]) / unperm_up
        if unperm_down > 0:
            conf_bins_down[idx] = (unperm_down - num_null_down[idx]) / unperm_down

    # TODO: Code like this was in the original PaGE, presumably to
    # ensure that the bins are monotonically increasing. Is this
    # necessary?
    for idx in np.ndindex(s, n):
        ensure_increases(conf_bins_up[idx])
        ensure_increases(conf_bins_down[idx])
    
    print "Computing confidence scores"
    (gene_conf_up, gene_conf_down) = get_gene_confidences(
        table, unperm_stats, mins, maxes, conf_bins_up, conf_bins_down)
    
    print "Counting up- and down-regulated features in each level"
    levels = np.linspace(0.5, 0.95, 10)

    (up_by_conf, down_by_conf) = get_count_by_conf_level(gene_conf_up, gene_conf_down, levels)

    breakdown = breakdown_tables(levels, up_by_conf, down_by_conf)
    logging.info("Levels are " + str(levels))
    return (conf_bins_up, conf_bins_down, breakdown)

def ensure_increases(a):
    for i in range(len(a) - 1):
        a[i+1] = max(a[i], a[i+1])

def breakdown_tables(levels, up_by_conf, down_by_conf):
    (num_range_values, n, num_levels) = np.shape(up_by_conf)

    max_up_params   = np.argmax(up_by_conf, axis=0)
    max_down_params = np.argmax(down_by_conf, axis=0)
 
    breakdown = np.zeros((n, len(levels), 3))

    for c in range(1, n):
            
        breakdown[c, :, 0] = levels
        for i in range(len(levels)):
            breakdown[c, i, 1] = up_by_conf[max_up_params[c, i], c, i]
            breakdown[c, i, 2] = down_by_conf[max_down_params[c, i], c, i]

    return breakdown

def print_counts_by_confidence(breakdown, condition_names):

    """Breakdown is an (n x levels x 3) table, where n is the number
    of conditions and levels is the number of confidence levels. It
    represents a list of tables, one for each condition, containing
    the confidence level, the number of up-regulated features, and the
    number of down-regulated features for each confidence level.
    """

    (n, levels, cols) = np.shape(breakdown)
    
    for c in range(1, n):
        print """
----------------------------
{:s}
{:10s} {:7s} {:7s}
----------------------------
""".format(str(condition_names[c]), 'confidence', 'num. up', 'num. down')

        for row in breakdown[c]:
            (level, up, down) = row
            print "{:10.2f} {:7d} {:9d}".format(level, int(up), int(down))


def get_count_by_conf_level(gene_conf_up, gene_conf_down, ranges):

    (num_range_values, num_genes, num_conditions) = np.shape(gene_conf_up)
    shape = (num_range_values, num_conditions, len(ranges))

    up_by_conf   = np.zeros(shape)
    down_by_conf = np.zeros(shape)
    
    for i in range(num_range_values):
        for j in range(num_conditions):
            up_conf   = gene_conf_up  [i, :, j]
            down_conf = gene_conf_down[i, :, j]
            for (k, level) in enumerate(ranges):
                up_by_conf  [i, j, k] = len(up_conf  [up_conf   > level])
                down_by_conf[i, j, k] = len(down_conf[down_conf > level])

    return (up_by_conf, down_by_conf)

def get_gene_confidences(table, unperm_stats, mins, maxes, conf_bins_up, conf_bins_down):
    """Returns a pair of 3D arrays: gene_conf_up and
    gene_conf_down. gene_conf_up[i, j, k] indicates the confidence
    with which gene j is upregulated in condition k using the ith
    alpha multiplier. gene_conf_down does the same thing for
    down-regulation."""

    (num_range_values, num_genes, num_conditions) = np.shape(unperm_stats)
    num_bins = np.shape(conf_bins_up)[2] - 1

    gene_conf_shape = (num_range_values, num_genes, num_conditions)
    gene_conf_up    = np.zeros(gene_conf_shape)
    gene_conf_down  = np.zeros(gene_conf_shape)

    for c in range(1, num_conditions):
        for i in range(num_range_values):
            for j in range(num_genes):
                if unperm_stats[i, j, c] >= 0:			
                    binnum = int(num_bins * unperm_stats[i, j, c] / maxes[i, c])
                    gene_conf_up[i, j, c] = conf_bins_up[i, c, binnum]
                else:
                    binnum = int(num_bins * unperm_stats[i, j, c] / mins[i, c])
                    gene_conf_down[i, j, c] = conf_bins_down[i, c, binnum]

    return (gene_conf_up, gene_conf_down)

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
    u_bins = get_bins(num_bins + 1, maxval)
    d_bins = get_bins(num_bins + 1, -minval)

    (u_hist, u_edges) = np.histogram(vals, u_bins)
    (d_hist, d_edges) = np.histogram( -vals, d_bins)
    u_hist[0] += len(vals[vals < 0.0])
    d_hist[0] += len(vals[vals > 0.0])

    return (u_hist, d_hist)

def dist_unpermuted_stats(table, conditions, mins, maxes, default_alphas, num_bins=1000):
    """
    Returns a tuple of three items, (up, down, stats). up is an (l x m
    x n) array where l is the number of tuning parameters, m is the
    number of conditions, and n is the number of bins. op[i,j,k] is
    the number of features that would be reported upregulated in
    condition i with tuning param j, in bin k. down is a similar array
    for downregulated features. stats is an (m x l) matrix where m is
    the number of features and l is the number of tuning parameters.
    """

    hist_shape = (len(TUNING_PARAM_RANGE_VALUES),
                  len(conditions),
                  num_bins + 1)

    u = np.zeros(hist_shape, dtype=int)
    d = np.zeros(hist_shape, dtype=int)

    center = 0

    stats = np.zeros((len(TUNING_PARAM_RANGE_VALUES),
                      len(table),
                      len(conditions)))
    
    for c in range(1, len(conditions)):

        alphas = default_alphas[c] * TUNING_PARAM_RANGE_VALUES

        v1 = table[:, conditions[0]]
        v2 = table[:, conditions[c]]
        stats[:, :, c] = tstat(v2, v1, alphas)

        for j in range(len(TUNING_PARAM_RANGE_VALUES)):
            (u_hist, d_hist) = assign_bins(stats[j, :, c], num_bins, mins[j, c], maxes[j, c])
            d[j, c, :] = d_hist
            u[j, c, :] = u_hist

    return (u, d, stats)
    

def get_bins(n, maxval):

    # Bin 0 in the "up" histogram is for features that were down-regulated
    bins = []
    bins.extend(np.linspace(0, maxval, n))

    # Bin "numbin" in the "up" histogram is for features that were
    # above the max observed in the unpermuted data
    bins.append(np.inf)
    return bins


    
