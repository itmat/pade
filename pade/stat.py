"""Low-level statistical methods.

This module should be general-purpose, and not have any dependencies
on the data model used in PADE or the workflow. The idea is that we
may use these functions outside of the standard PADE workflow.

"""

from __future__ import absolute_import, print_function, division

import itertools
import logging
import numpy as np

from scipy.stats import gmean
from bisect import bisect
from pade.layout import (
    intersect_layouts, apply_layout, layout_is_paired)

class UnsupportedLayoutException(Exception):
    """Thrown when a statistic is used with a layout that it can't support."""
    pass

def double_sum(data):
    """Returns the sum of data over the last two axes."""
    return np.sum(np.sum(data, axis=-1), axis=-1)


def group_means(data, layout):
    """Get the means for each group defined by layout.

    Groups data according to the given layout and returns a new
    ndarray with the same number of dimensions as data, but with the
    last dimension replaced by the means according to the specified
    grouping.
    
    One dimensional input:

    >>> group_means(np.array([-1, -3, 4, 6]), [[0, 1], [2, 3]])
    array([-2.,  5.])

    Two dimensional input:

    >>> data = np.array([[-1, -3, 4, 6], [10, 12, 30, 40]])
    >>> layout = [[0, 1], [2, 3]]
    >>> group_means(data, layout) # doctest: +NORMALIZE_WHITESPACE
    array([[ -2.,  5.],
           [ 11., 35.]])

    :param data: An ndarray. Any number of dimensions is allowed.

    :param layout: A :term:`layout` describing the data.

    :return: An ndarray giving the means for each group obtained by
      applying the given layout to the given data.

    """
    # We'll take the mean of the last axis of each group, so change
    # the shape of the array to collapse the last axis down to one
    # item per group.
    shape = np.shape(data)[:-1] + (len(layout),)
    res = np.zeros(shape)

    for i, group in enumerate(apply_layout(data, layout)):
        res[..., i] = np.mean(group, axis=-1)

    return res

def residuals(data, layout):
    """Return the residuals for the given data and layout.

    >>> residuals(np.array([1, 2, 3, 6], float), [[0, 1], [2, 3]])
    array([-0.5,  0.5, -1.5,  1.5])

    :param data: 
      An ndarray. Any number of dimensions is allowed.

    :param layout:
      A :term:`layout` describing the data.

    :return: 
      The residuals obtained by subtracting the means of the groups
      defined by the layout from the values in data.

    """
    means = group_means(data, layout)
    diffs = []
    groups = apply_layout(data, layout)
    
    for i, group in enumerate(groups):
        these_means = means[..., i].reshape(np.shape(group)[:-1] + (1,))
        diffs.append(group - these_means)
    return np.concatenate(diffs, axis=-1)

def rss(data, layout=None):
    """Return the residual sum of squares for the data and optional layout.

    :param data:
      An n-dimensional array.

    :param layout:
      If provided, the means will becalculated based on the grouping
      given by the layout applied to the last axis of data. Otherwise,
      no grouping will be used.

    >>> rss(np.array([1, 2, 3, 6], float), [[0, 1], [2, 3]])
    5.0

    """

    if layout is None:
        y   = np.mean(data, axis=-1).reshape(np.shape(data)[:-1] + (1,))
        return double_sum((data  - y)  ** 2)

    else:
        r = residuals(data, layout)
        rs = r ** 2
        return np.sum(rs, axis=-1)

class LayoutPairTest(object):
    """Base class for a statistic that needs a pair of layouts."""

    def __init__(self, condition_layout, block_layout):
        self.condition_layout = condition_layout
        self.block_layout = block_layout


class Ftest(LayoutPairTest):
    """Computes the F-test.

    Some sample data

    >>> a = np.array([1., 2.,  3., 6.])
    >>> b = np.array([2., 1.,  1., 1.])
    >>> c = np.array([3., 1., 10., 4.])

    The condition layout has the first two columns in one group and
    the second two in another. There is no blocking, so the block
    layout has all columns in one group.

    >>> condition_layout = [[0, 1], [2, 3]]
    >>> block_layout     = [[0, 1, 2, 3]]
    
    Construct one ftest based on our layouts

    >>> ftest = Ftest(condition_layout, block_layout)
    
    Test one row

    >>> round(ftest(a), 1)
    3.6

    Test multiple rows at once

    >>> data = np.array([a, b, c])
    >>> ftest(data)
    array([ 3.6,  1. ,  2.5])

    """

    name = "F-test"

    def __init__(self, condition_layout, block_layout, alphas=None):

        super(Ftest, self).__init__(condition_layout, block_layout)

        full_layout = intersect_layouts(block_layout, condition_layout)
        if min(map(len, full_layout)) < 2:
            raise UnsupportedLayoutException(
                "I can't use an FTest with the specified layouts, because " +
                "the intersection between those layouts results in some " +
                "groups that contain fewer than two samples.")

        self.layout_full = full_layout
        self.alphas = alphas

    def __call__(self, data):

        # Degrees of freedom
        p_red  = len(self.block_layout)
        p_full = len(self.layout_full)
        n      = sum(map(len, self.block_layout))

        # Means and residual sum of squares for the reduced and full
        # model
        rss_full = rss(data, self.layout_full)
        rss_red  = rss(data, self.block_layout)

        numer = (rss_red - rss_full) / (p_full - p_red)
        denom = rss_full / (n - p_full)

        if self.alphas is not None:
            denom = np.array([denom + x for x in self.alphas])
        return numer / denom


class OneSampleTTest:
    def __init__(self, alphas=None):
        self.alphas = alphas

    def __call__(self, data):
        n = np.size(data, axis=-1)
        x = np.mean(data, axis=-1)
        s = np.std(data, axis=-1)

        numer = x
        denom = s / np.sqrt(n)
        if self.alphas is not None:
            denom = np.array([denom + x for x in self.alphas])
        return np.abs(numer / denom)


class MeansRatio(LayoutPairTest):

    """Means ratio statistic.

    Supports layouts where there are two experimental conditions, with
    or without blocking.

    :param condition_layout:
      A layout that groups the sample indexes together into groups
      that have the same experimental condition. MeansRatio only
      supports designs where there are exactly two conditions, so
      len(condition_layout) must be 2.

    :param block_layout: 
      If the input has blocking variables, then block layout
      should be a layout that groups the sample indexes together
      by block.

    :param alphas: 
      Optional array of "tuning parameters". 

    :param symmetric:
      If true, gives the inverse of the ratio when the ratio is less
      than 1. Use this when it does not matter which condition is
      greater than the other one.
      
    """

    name = "means ratio"

    def __init__(self, condition_layout, block_layout, alphas=None, symmetric=True):

        super(MeansRatio, self).__init__(condition_layout, block_layout)
        conditions = len(condition_layout)
        blocks     = len(block_layout)

        if conditions != 2:
            raise UnsupportedLayoutException(
                ("MeansRatio only supports configurations where there are " +
                 "two conditions and n blocks. You have {conditions} " +
                 "conditions and {blocks} blocks.").format(
                    conditions=conditions,
                    blocks=blocks))

        self.alphas    = alphas
        self.symmetric = symmetric


    def __call__(self, data):

        conds  = self.condition_layout
        blocks = self.block_layout

        # Build two new layouts. c0 is a list of lists of indexes into
        # the data that represent condition 0 for each block. c1 is
        # the same for data that represent condition 1 for each block.
        c0_blocks = intersect_layouts(blocks, [ conds[0] ])
        c1_blocks = intersect_layouts(blocks, [ conds[1] ])

        # Get the mean for each block for both conditions.
        means0 = group_means(data, c0_blocks)
        means1 = group_means(data, c1_blocks)

        # If we have tuning params, add another dimension to the front
        # of each ndarray to vary the tuning param.
        if self.alphas is not None:
            shape = (len(self.alphas),) + np.shape(means0)
            old0 = means0
            old1 = means1
            means0 = np.zeros(shape)
            means1 = np.zeros(shape)
            for i, a in enumerate(self.alphas):
                means0[i] = old0 + a
                means1[i] = old1 + a

        means0 /= means1
        ratio = means0

        # If we have more than one block, we combine their ratios
        # using the geometric mean.
        ratio = gmean(ratio, axis=-1)

        # 'Symmetric' means that the order of the conditions does not
        # matter, so we should always return a ratio >= 1. So for any
        # ratios that are < 1, use the inverse.
        if self.symmetric:
            # Add another dimension to the front where and 1 is its
            # inverse, then select the max across that dimension
            ratio_and_inverse = np.array([ratio, 1.0 / ratio])
            ratio = np.max(ratio_and_inverse, axis=0)

        return ratio
        

class OneSampleDifferenceTTest(LayoutPairTest):
    """A one-sample t-test where the input is given as pairs.

    Input with two features (one on each row), eight samples
    arranged as four pairs.

    >>> table = np.array([[3, 2, 6, 4, 9, 6, 7, 3], [2, 4, 4, 7, 5, 1, 8, 3]])

    Pairs are grouped together. Assume we have two conditions, the
    even numbered samples are one condition and the odd numbered ones
    are the other

    >>> block_layout     = [ [0, 1], [2, 3], [4, 5], [6, 7] ]
    >>> condition_layout = [ [0, 2, 4, 6], [1, 3, 5, 7] ]

    Construct the test function with the condition and block layouts.

    >>> test = OneSampleDifferenceTTest(condition_layout, block_layout)

    Apply it to 1d input (the first feature in the table):

    >>> round(test(table[0]), 7)
    4.472136

    Now 2d input (both features in the table):

    >>> results = test(table)
    >>> round(results[0], 7)
    4.472136
    >>> round(results[1], 7)
    0.5656854

    """
    name = "OneSampleDifferenceTTest"

    def __init__(self, condition_layout, block_layout, alphas=None):
        super(OneSampleDifferenceTTest, self).__init__(condition_layout, block_layout)

        if not layout_is_paired(block_layout):
            raise UnsupportedLayoutException(
                "The block layout " + str(block_layout) + " " +
                "is invalid for a one-sample difference t-test. " +
                "Each block must be a pair, with exactly two items in it")
        
        if len(condition_layout) != 2:
            raise UnsupportedLayoutException(
                "The condition layout " + str(condition_layout) + " " +
                "is invalid for a one-sample difference t-test. " +
                "There must be two conditions, and you have " + 
                str(len(condition_layout)) + ".")

        self.child = OneSampleTTest(alphas)

    def __call__(self, data):
        
        pairs = self.block_layout
        conds = self.condition_layout
        values = []

        for i in [ 0, 1 ]:
            # Make a new layout that is just the item for each pair
            # from condition i. layout will be a list of sets, each
            # with just one index, since there is only one item from
            # each pair with condition i. So flatten it into a list of
            # indexes, and grab the corresponding values from the
            # data.
            layout = intersect_layouts(self.block_layout, [ conds[i] ])
            idxs = list(itertools.chain(*layout))
            values.append(data[..., idxs])

        # Now just get the differences between the two sets of values
        # and call the child statistic on those values.
        return self.child(values[0] - values[1])

### Code review stops here


class GroupSymbols(object):

    def __init__(self, condition_layout):
        self.condition_layout = condition_layout

        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

        self.index_to_symbol = {}
        for i, grp in enumerate(condition_layout):
            for idx in grp:
                self.index_to_symbol[idx] = letters[i]

    def __call__(self, data):
        groups = apply_layout(data, self.condition_layout)
        res = []
        for i, grp in enumerate(groups):
            syms = []
            for index in grp:
                sym = self.index_to_symbol[index]
                syms.append(sym)

            res.append(''.join(sorted(syms)))


        return ' '.join(res)

            
def bootstrap(data,
              stat_fn,
              R=1000,
              layout=None,
              permutations=None,
              residuals=None,
              bins=None):
    """Run bootstrapping.

    This function should most likely accept data of varying
    dimensionality, but this documentation uses two dimensional data
    as an example.

    :param data:
      An (M x N) array.

    :param stat_fn:
      A callable that accepts an array of shape (M x N) and returns
      statistics of shape (M).

    :param R:
      The number of bootstrapping samples to generate, if
      *permutations* is not supplied.

    :param layout:
      If *permutations* is not supplied, layout can be used to
      specify a :term:`layout` to restrict the randomized sampling. If
      supplied, it must be a list of lists which divides the N indexes
      of the columns of *data* up into groups.

    :param permutations:
      If supplied, it must be an (M x N) table of indexes into the
      data, which we will use to extract data points for
      bootstrapping. If not supplied, *R* indexes will be generated
      randomly, optionally restricted by the :layout: given in
      *layout*.

    :param residuals:
      You can use this in conjunction with the *data* parameter to
      construct artificial samples. If this is supplied, it must be
      the same shape as *data*. Then *data* should be the values
      predicted by the model, and *residuals* should be the residuals
      representing the predicted values subtracted from the original
      data. The samples will be constructed by selecting random
      samples of the residuals and adding them back onto *data*. So if
      residuals are provided, then *data + residuals* should be equal
      to the original, raw data.

    :param bins:
      An optional list of numbers representing the edges of bins into
      which we will accumulate mean counts of statistics.

    :return:
      If *bins* is not provided, I will return an :math:`(R x M)`
      array giving the value of the statistic for each row of *data*
      for sample.

      If *bins* is provided, I will return a list of length
      :math:`len(bins) - 1` where each item is the average number of
      rows of *data* across all samples that have statistic value
      falling in the range associated with the corresponding bin.

      """

    if residuals is None:
        build_sample = lambda idxs: data[..., idxs]
    else:
        build_sample = lambda idxs: data + residuals[..., idxs]

    if permutations is None:
        if layout is None:
            layout = [ np.arange(np.shape(data)[1]) ]
        permutations = random_indexes(layout, R)

    # If we did not get bins, we simply return an ndarray of all the
    # statistics we got. So initialize the result to [], reduce it by
    # just appending the new result to the table, and finalize it by
    # turning it into an ndarray.
    if bins is None:
        initial_value = []
        reduce_fn = lambda res, val: res + [ val ],
        finalize_fn = lambda x: np.array(x)

    # If we got bins, we want to accumulate counts into those bins and
    # then take the average by dividing the count in each bins by the
    # number of permutations.
    else:
        initial_value = np.zeros(cumulative_hist_shape(bins))
        reduce_fn = lambda res, val : res + cumulative_hist(val, bins)
        finalize_fn = lambda res : res / len(permutations)

    # We'll return an R x n array, where n is the number of
    # features. Each row is the array of statistics for all the
    # features, using a different random sampling.
    
    samples = (build_sample(p) for p in permutations)
    stats   = (stat_fn(s)      for s in samples)

    reduced = reduce(reduce_fn, stats, initial_value)
    res = finalize_fn(reduced)

    return res

def cumulative_hist_shape(bins):
    """Returns the shape of the histogram with the given bins.

    The shape is similar to that of bins, except the last dimension
    has one less element.

    """
    shape = np.shape(bins)
    return shape[:-1] + (shape[-1] - 1,)

def cumulative_hist(values, bins):
    """Create a cumulative histogram for values using the given bins.

    The shape of values and bins must be the same except for the last
    dimension.  So np.shape(values)[:-1] must equal
    np.shape(bins[:-1]). The last dimension of values is simply a
    listing of values. The last dimension of bins is the list of bin
    edges for the histogram.

    """
    shape = cumulative_hist_shape(bins) 
    res = np.zeros(shape)
    for idx in np.ndindex(shape[:-1]): 
        (hist, ignore) = np.histogram(values[idx], bins[idx]) 
        res[idx] = np.array(np.cumsum(hist[::-1])[::-1], float)
    return res


def bins_uniform(num_bins, stats):
    """Returns a set of evenly sized bins for the given stats.

    Stats should be an array of statistic values, and num_bins should
    be an integer. Returns an array of bin edges, of size num_bins +
    1. The bins are evenly spaced between the smallest and largest
    value in stats.

    Note that this may not be the best method for binning the
    statistics, especially if the distribution is heavily skewed
    towards one end.

    """
    base_shape = np.shape(stats)[:-1]
    bins = np.zeros(base_shape + (num_bins + 1,))
    for idx in np.ndindex(base_shape):
        maxval = np.max(stats[idx])
        edges = np.concatenate((np.linspace(0, maxval, num_bins), [np.inf]))
        edges[0] = - np.inf
        bins[idx] = edges

    return bins


def bins_custom(num_bins, stats):
    """Get an array of bin edges based on the actual computed
    statistic values. stats is an array of length n. Returns an array
    of length num_bins + 1, where bins[m, n] and bins[m + 1, n] define
    a bin in which to count features for condition n. There is a bin
    edge for negative and positive infinity, and one for each
    statistic value.

    """
    base_shape = np.shape(stats)[:-1]
    bins = np.zeros(base_shape + (num_bins + 1,))
    bins[ : -1] = sorted(stats)
    bins[-1] = np.inf
    return bins


def confidence_scores(raw_counts, perm_counts, num_features):
    """Return confidence scores.
    
    """
    logging.debug(("Getting confidence scores for shape {shape} with "
                   "{num_features} features").format(
            shape=np.shape(raw_counts),
            num_features=num_features))
    if np.shape(raw_counts) != np.shape(perm_counts):
        raise Exception((
                "raw_counts and perm_counts must have same shape. "
                "raw_counts is {raw} and perm_counts is {perm}").format(
                raw=np.shape(raw_counts), perm=np.shape(perm_counts)))
    
    shape = np.shape(raw_counts)
    adjusted = np.zeros(shape)
    for idx in np.ndindex(shape[:-1]):
        adjusted[idx] = adjust_num_diff(perm_counts[idx], raw_counts[idx], num_features)

    # (unpermuted counts - mean permuted counts) / unpermuted counts
    res = (raw_counts - adjusted) / raw_counts

    for idx in np.ndindex(res.shape[:-1]):
        res[idx] = ensure_scores_increase(res[idx])

    return res


def assign_scores_to_features(stats, bins, scores):
    """Return an array that gives the confidence score for each feature.
    
    :param stats:
      An array giving the statistic value for each feature.

    :param bins: 
      A monotonically increasing array which divides the statistic
      space up into ranges.

    :param scores:
      A monotonically increasing array of length (len(bins) - 1) where
      scores[i] is the confidence level associated with statistics
      that fall in the range (bins[i-1], bins[i]).

    :return:
      An array that gives the confidence score for each feature.

    """
    logging.debug(("I have {num_stats} stats, {num_bins} bins, and " +
                  "{num_scores} scores").format(num_stats=np.shape(stats),
                                                num_bins=np.shape(bins),
                                                num_scores=np.shape(scores)))

    shape = np.shape(stats)
    res = np.zeros(shape)

    for idx in np.ndindex(shape):
        prefix = idx[:-1]
        stat = stats[idx]
        scores_idx = prefix + (bisect(bins[prefix], stat) - 1,)
        res[idx] = scores[scores_idx]
    logging.debug("Scores have shape {0}".format(np.shape(res)))
    return res


def adjust_num_diff(V0, R, num_ids):
    V = np.zeros((6,) + np.shape(V0))
    V[0] = V0
    for i in range(1, 6):
        V[i] = V[0] - V[0] / num_ids * (R - V[i - 1])
    return V[5]

def ensure_scores_increase(scores):
    """Returns a copy of the given ndarray with monotonically increasing values.

    """
    res = np.copy(scores)
    for i in range(1, len(res)):
        res[i] = max(res[i], res[i - 1])
    return res
