#!/usr/bin/env python

import argparse
import re
import numpy as np
from numpy import std, arange, sqrt, mean, ma, shape, size, inf, linspace, histogram
import itertools 

from scipy.misc import comb

__version__ = '6.0.0'

stat_tstat = 0
stat_means = 1
stat_medians = 2

class Config:
    def __init__(self, args):
        self.num_channels = None
        self.infile = None

        if 'num_channels' in args:
            self.num_channels = args.num_channels
        if 'infile' in args:
            self.infile = args.infile

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return "Config(" + repr(self.__dict__) + ")"

class Input:
    def __init__(self, row_ids, col_ids, table):
        self.row_ids    = row_ids
        self.column_ids = col_ids
        self.table      = ma.masked_array(table, np.zeros(shape(table)))
#        self.table      = table
        self.column_indices = {}
        self.num_bins = 1000

        pat = re.compile("c(\d+)r(\d+)")
        columns = []
        counter = 0
        for s in col_ids:
            m = pat.match(s)
            if m is None:
                raise Exception("Bad column id " + s)
            c = int(m.group(1))
            r = int(m.group(2))

            while len(columns) <= c:
                columns.append([])
            while len(columns[c]) <= r:
                columns[c].append(None)
            columns[c][r] = counter
            counter += 1
        self.columns = columns

    def num_conditions(self):
        return len(self.columns)

    def replicates(self, condition):
        return sorted(self.columns[condition])[1:]

    def conditions(self):
        res = []
        for c in range(self.num_conditions()):
            res.append(self.replicates(c))
        return res

def show_banner():
    print """
------------------------------------------------------------------------------

                       Welcome to PaGE {version}
                   Microarray and RNA-Seq analysis tool

------------------------------------------------------------------------------

For help, please use {script} --help or {script} -h
""".format(version=__version__,
           script=__file__)

def get_arguments():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    file_locations = parser.add_argument_group("File locations")

    file_locations.add_argument(
        '--infile',
        help="""Name of the input file containing the table of
        data. Must conform to the format in the README file.""",
        default=argparse.SUPPRESS,
        type=file)

    file_locations.add_argument(
        '--outfile',
        help="""Name of the output file. If not specified outfile name will be
        derived from the infile name.""",
        default=argparse.SUPPRESS,
        type=file)

    file_locations.add_argument(
        "--id2info",
        help="""Name of the file containing a mapping of gene id's to
        names or descriptions.""",
        default=argparse.SUPPRESS,
        type=file)

    file_locations.add_argument(
        "--id2url",  
        help="""Name ot the file containing a mapping of gene id's to urls.""",
        default=argparse.SUPPRESS,
        type=file)

    file_locations.add_argument(
        "--id-filter-file", 

        help="""If you just want to run the algorithm on a subset of
        the genes in your data file, you can put the id's for those
        genes in a file, one per line, and specify that file with
        this option.""",
        default=argparse.SUPPRESS,
        type=file)

    output_options = parser.add_argument_group("Output options")

    output_options.add_argument(
            "--output-gene-confidence-list",

            help="""Set this to output a tab delimited file that maps
            every gene to its confidence of differential expression.
            For each comparison gives separate lists for up and down
            regulation. """,
            default=argparse.SUPPRESS,
            type=file)

    output_options.add_argument(
            "--output-gene-confidence-list-combined",

            help="""Set this to output a tab delimited file that maps
            every gene to its confidence of differential expression.
            For each comparison gives one list with up and down
            regulation combined. """,
            default=argparse.SUPPRESS,
            type=file)

    output_options.add_argument(
            "--output-text",
            help="""Set this to output the results also in text format. """,
            default=argparse.SUPPRESS,
            action='store_true')

    output_options.add_argument(
            "--note",
            default=argparse.SUPPRESS,
            help="""A string that will be included at the top of the
            output file.""")

    output_options.add_argument(
            "--aux-page-size", 
            type=int,
            default=500,
            help="""A whole number greater than zero.  This specifies
            the minimum number of tags there can be in one pattern
            before the results for that pattern are written to an
            auxiliary page (this keeps the main results page from
            getting too large).""")

    design = parser.add_argument_group(
        "Study design and nature of the input data")

    design.add_argument(
        "--num-channels",
        type=int,
        default=argparse.SUPPRESS,        
        choices=[1,2],
        help="""Is your data one or two channels?  (note: Affymetrix
        is considered one channel).""")

    design.add_argument(
        "--design", 
        default=argparse.SUPPRESS,
        choices=['d', 'r'],
        help="""For two channel data, either set this to "r" for
        "reference" design or "d" for "direct comparisons"
        (see the documentation for more  information on this
        setting). """)

    design.add_argument(
        "--data-is-logged", 
        default=argparse.SUPPRESS,
        action='store_const',
        dest='data_is_logged',
        const=True,
        help="Use this option if your data has already been log transformed.")

    design.add_argument(
        """--data-not-logged""",
        action='store_const',
        dest='data_is_logged',
        const=False,
        default=argparse.SUPPRESS,
        help="Use this option if your data has not been log transformed. ")

    paired = design.add_mutually_exclusive_group()

    paired.add_argument(
        "--paired", 
        action='store_const',
        dest='paired',
        const=True,
        default=argparse.SUPPRESS,
        help="The data is paired.")

    paired.add_argument(
        "--unpaired", 
        action='store_const',
        dest='paired',
        const=False,
        default=argparse.SUPPRESS,
        help="The data is not paired.")

    design.add_argument(
        "--missing-value", 
        default=argparse.SUPPRESS,
        help="""If you have missing values designated by a string
        (such as \"NA\"), specify  that string with this option.  You
        can either put quotes around the string  or not, it doesn't
        matter as long as the string has no spaces."""
)

    stats = parser.add_argument_group('Statistics and parameter settings')

    stats.add_argument(
        "--level-confidence",
        type=float,
        default=argparse.SUPPRESS,
        help="""A number between 0 and 1. Generate the levels with
        this confidence. See the README file for more information on
        this parameter. This can be set separately for each group
        using --level-confidence-list (see below).  NOTE: This
        parameter can be set at the end of the run after the program
        has displayed a summary breakdown of how many genes are found
        with what confidence. To do this either set the command line
        option to "L" (for "later"), or do not specify this command
        line option and enter "L" when the program prompts for the
        level confidence.""")

    stats.add_argument(
        "--level-confidence-list",
        default=argparse.SUPPRESS,
        help="""Comma-separated list of confidences. If there are more
        than two conditions (or more than one direct comparision),
        each position in the pattern can have its own confidence
        specified by this list. E.g. if there are 4 conditions, the
        list might be .8,.7,.9 (note four conditions gives patterns of
        length 3)""")

    stats.add_argument(
        "--min-presence",
        default=argparse.SUPPRESS,
        help="""A positive integer specifying the minimum number of
        values a tag should  have in each condition in order to not
        be discarded.  This can be set  separately for each condition
        using --min-presence-list """)

    stats.add_argument(
        "--min-presence-list",
        default=argparse.SUPPRESS,
        help="""Comma-separated list of positive integers, one for
        each condition,  specifying the minimum number of values a
        tag should have, for each  condition, in order not to be
        discarded.  E.g. if there are three  conditions, the list
        might be 4,6,3 """)

    stats.add_argument(
        "--use-logged-data",
        default=argparse.SUPPRESS,
        help="""Use this option to run the algorithm on the logged
        data (you can only  use this option if using the t-statistic
        as statistic).  Logging the  data usually give better
        results, but there is no rule.  Sometimes  different genes
        can be picked up either way.  It is generally best,  if using
        the t-statistic, to go with the logged data.  You might try 
        both ways and see if it makes much difference.  Both ways give
        valid  results, what can be effected is the power. """)

    stats.add_argument(
        "--use-unlogged-data",
        default=argparse.SUPPRESS,
        help="""Use this option to run the algorithm on the unlogged
        data.  (See  --use-loggged-data option above for more
        information.) """)

    stats.add_argument(
        "--tstat",
        default=argparse.SUPPRESS,
        help="Use the t-statistic as statistic. ")

    stats.add_argument(
        "--means",
        default=argparse.SUPPRESS,
        help="Use the ratio of the means of the two groups as statistic. ")

    stats.add_argument(
        "--tstat-tuning-parameter",
        default=argparse.SUPPRESS,
        help="""Optional.  The value of the t-statistic tuning
        parameter.  This is set to  a default value determined
        separately for each pattern position, but can be  set by hand
        using this command.  See the documentation for more 
        information on this parameter. """)

    stats.add_argument(
        "--shift",
        default=argparse.SUPPRESS,
        help="""Optional.  A real number greater than zero.  This
        number will be added to  all intensities (of the unlogged
        data).  See the documentation for more on  why you might use
        this parameter. """)

    return parser.parse_args()
    
    
def main():

    show_banner()
    args = get_arguments()
    config = validate_args(args)
    data = load_input(config)
    alphas = find_default_alpha(data.table, data.conditions())
    do_confidences_by_cutoff(data, alphas)
    
    print config

def validate_args(args):

    c = Config(args)

    pos_int_re = re.compile("\d+")

    if 'num_channels' in args:
        if args.num_channels == 1 and 'design' in args:
            raise Exception("Error: if the number of channels is 1, do not specify the design type\n\n")
    elif 'design' in args:
        c.num_channels = 2

    while c.num_channels is None:
        s = raw_input("Are the arrays 1-channel or 2-channel arrays? (Enter 1 or 2): ")
        if pos_int_re.match(s) is not None:
            channels = int(s)

            if channels == 1 or channels == 2:
                c.num_channels = channels

    return c


def load_input(config):
    f = config.infile
    if type(f) == str:
        f = open(f, 'r')

    headers = f.next().rstrip().split("\t")

    ids = []
    table = []

    for line in f:
        row = line.rstrip().split("\t")
        rowid = row[0]
        values = [float(x) for x in row[1:]]
        ids.append(rowid)
        table.append(values)

    table = np.array(table)

    return Input(ids, headers[1:], table)

def unpermuted_means(data):
    num_conditions = data.num_conditions()
    num_features   = len(data.row_ids)

    res = np.zeros((num_features, num_conditions))

    for c in range(num_conditions):
        cols = data.replicates(c)
        print "Computing mean for condition %d using replicates %s" % (c, cols)
        cols = data.table[:,cols]
        means = mean(cols, axis=1)
        print "%s -> %s" % (shape(cols), shape(means))
        res[:,c] = means
    return res

def compute_s(v1, v2, mp1, mp2, axis=0):
    """
    v1 and v2 should have the same number of rows.
    """

    sd1 = std(v1, ddof=1, axis=axis)
    sd2 = std(v2, ddof=1, axis=axis)
    
    s1 = size(v1, axis=axis) - 1
    s2 = size(v2, axis=axis) - 1

    return sqrt((sd1 ** 2 * s1 +
                 sd2 ** 2 * s2)  / (s1 + s2))

def find_default_alpha(table, conditions):

    baseline_cols = conditions[0]
    baseline_data = table[:,baseline_cols]

    alphas = np.zeros(len(conditions))

    for (c, cols) in enumerate(conditions):
        if c == 0: 
            continue

        condition_data = table[:,cols]
        
        values = compute_s(condition_data, baseline_data, None, None, axis=1)

        the_mean = mean(values)

        lt_mean = values[values < the_mean]
        residuals = lt_mean - the_mean

        sd = sqrt(sum(residuals ** 2) / (len(residuals) - 1))

        alphas[c] = the_mean * 2 / sqrt(len(cols) + len(baseline_cols))

    return alphas


tuning_param_range_values = np.array([
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


def v_tstat(v1, v2, alphas, axis=0):
    """
    Computes the t-statistic for two 2-D arrays. v1 is an m x n1 array
    and v2 is an m x n2 array, where m is the number of features, n1
    is the number of replicates in the condition represented by v1,
    and n2 is the number of conditions for v2. Returns an (m x p)
    array, where m again is the number of features, and p is the
    number of values in the alphas array.
    """

    # Standard deviations of v1 and v2 with one degree of freedom
    sd1 = std(v1, ddof=1, axis=axis)
    sd2 = std(v2, ddof=1, axis=axis)

    len1 = np.array([len(row) for row in v1])
    len2 = np.array([len(row) for row in v2])

    S = sqrt((sd1**2*(len1-1) + sd2**2*(len2-1))/(len1 + len2 - 2))

    result = np.zeros((len(alphas), len(v1)))
    numer  = (mean(v1, axis=axis) - mean(v2, axis=axis)) * sqrt(len1 * len2)

    for i in range(0, len(alphas)):
        rhs = numer / ((alphas[i] + S) * sqrt(len1 + len2))
        result[i,:] = rhs

    return result

def all_subsets(n, k):

    """
    
    Return an (m x n) matrix where n is the size of the set, n is the
    number of subsets of
    """

    indexes = arange(n)
    m = comb(n, k)
    result = np.zeros((m, n), dtype=bool)

    i = 0

    for i, subset in enumerate(itertools.combinations(indexes, k)):
        for j in subset:
            result[i,j] = True

    return result

def init_perms(data):
    perms = [None]

    baseline_len = len(data.replicates(0))

    for c in range(1, data.num_conditions()):
        this_len = len(data.replicates(c))
        n = baseline_len + this_len
        k = min(baseline_len, this_len)
        perms.append(all_subsets(n, k))

    return perms

def min_max_stat(data, conditions, default_alphas):
    """
    Returns a tuple (mins, maxes) where both mins and maxes are (m x
    n) matrices, m being the length of default_alphas, and n being the
    number of conditions.
    """

    table = np.zeros((len(conditions),
                   len(tuning_param_range_values), 
                   len(data)))

    for j in range(1, len(conditions)):
        alphas = default_alphas[j] * tuning_param_range_values

        table[j,:,:] = v_tstat(data[:,conditions[j]],
                               data[:,conditions[0]],
                               alphas,
                               axis=1)
        
    mins  = np.min(table, axis=2)
    maxes = np.max(table, axis=2)

    return (mins, maxes)

def dimsum(a, axis):
    res = cumsum(a[:,:,::-1], axis=axis)[:,:,::-1]
    return res

def do_confidences_by_cutoff(
    data, 
    default_alphas,
    ):
    all_perms = init_perms(data)

    base_len = len(data.replicates(0))
    for c in range(1, data.num_conditions()):
        print 'Working on condition %d of %d' % (c, data.num_conditions() - 1)
        perms = all_perms[c]

        # This is the list of all indexes into data.table for
        # replicates of condition 0 and condition c.
        master_indexes = []
        master_indexes.extend(data.replicates(0))
        master_indexes.extend(data.replicates(c))
        master_indexes = np.array(master_indexes)

        l = len(perms)
        m = len(data.row_ids)
        n = len(master_indexes)
        n2 = len(tuning_param_range_values)

        permuted_data = np.zeros((l, m, n))

        permuted_indexes = np.zeros((l, n), dtype=int)

        stats = np.zeros((l, len(tuning_param_range_values), len(data.table)))

        print "  Permuting indexes"
        for perm_num, perm in enumerate(perms):
            permuted_indexes[perm_num,0:base_len] = master_indexes[perm]
            permuted_indexes[perm_num,base_len:]  = master_indexes[~perm]

        print "  Permuting data"
        for perm_num, perm in enumerate(perms):
            permuted_data[perm_num, :] = data.table[:, permuted_indexes[perm_num]]

        print "  Getting stats"
        for perm_num, perm in enumerate(perms):
            v1 = permuted_data[perm_num, : , : base_len]
            v2 = permuted_data[perm_num, : , base_len :]
            stats[perm_num, : ] = v_tstat(v2, v1, default_alphas[c] * tuning_param_range_values, axis=1)
        (mins, maxes) = min_max_stat(data.table, data.conditions(), default_alphas)

        up   = np.zeros((l, n2, data.num_bins + 1), int)
        down = np.zeros((l, n2, data.num_bins + 1), int)

        print "  Counting up- and down-regulated features"

        print "Getting perms"
        for perm_num, perm in enumerate(perms):
            for i in range(len(tuning_param_range_values)):
                up_bins   = get_bins(data.num_bins, maxes[i, c])
                down_bins = get_bins(data.num_bins, -mins[i, c])
                vals      = stats[perm_num, :, i]
                (u_hist, u_edges) = histogram(vals, bins=up_bins)
                (d_hist, d_edges) = histogram(-vals, bins=down_bins)
                up  [perm_num, i] = u_hist
                down[perm_num, i] = d_hist

        print "Done"
        num_unpooled_up_vect   = np.zeros((n2, data.num_conditions(), data.num_bins + 1), int)
        num_unpooled_down_vect = np.zeros((n2, data.num_conditions(), data.num_bins + 1), int)

        # Bin 0 is for features that were downregulated (-inf, 0)
        # Bins 1 through 999 are for features that were upregulated
        # Bin 1000 is for any features that were upregulated above the max from the unmpermuted data (max, inf)

        up   = dimsum(up, axis=2)
        down = dimsum(down, axis=2)

#        for i in range(l):
#            for j in range(n2):
#                for k in range(data.num_bins + 1):
#                    print "up[%d][%d][%d] = %f" % (i, j, k, up[i, j, k])

        for perm_num, perm in enumerate(perms):
            for j, param in enumerate(tuning_param_range_values):
                num_unpooled_up_vect[j, c]   += up[perm_num, j]
                num_unpooled_down_vect[j, c] += down[perm_num, j]

 #       print "Up is " + str(up)

        mean_perm_up_vect   = num_unpooled_up_vect   / float(l)
        mean_perm_down_vect = num_unpooled_down_vect / float(l)

    (num_unperm_up, num_unperm_down, unperm_stats) = dist_unpermuted_stats(data, mins, maxes, default_alphas)
    
    print "Num unperm up is " + str(num_unperm_up)

#        for j in range(len(tuning_param_range_values)):
#            for i in range(data.num_bins + 1):
#                print "mean_perm_up_vect[%d][%d][%d]= %f" % (j, c, i, mean_perm_up_vect[j, c, i])
#            for i in range(data.num_bins + 1):
#                print "mean_perm_down_vect[%d][%d][%d]= %f" % (j, c, i, mean_perm_down_vect[j, c, i])
        
#        for j in range(n2):
#            for i in range(data.num_bins + 1):
#                print "up[%d][%d] = %d" % (j, i, up[0, j, i])

        #print permuted_data[0, 0]


def assign_bins(vals, num_bins, minval, maxval):
    """
    Computes two histograms for the given values.
    """
    u_bins = get_bins(num_bins + 1, maxval)
    d_bins = get_bins(num_bins + 1, -minval)

    (u_hist, u_edges) = histogram(vals, u_bins)
    (d_hist, d_edges) = histogram( -vals, d_bins)
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
    m = len(conditions)
    n = num_bins + 1

    l = len(tuning_param_range_values)

    u = np.zeros((l, m, n), dtype=int)
    d = np.zeros((l, m, n), dtype=int)

    center = 0

    stats = np.zeros((len(tuning_param_range_values),
                      len(table),
                      len(conditions)))
    
    for c in range(1, len(conditions)):

        alphas = default_alphas[c] * tuning_param_range_values

        v1 = table[:, conditions[0]]
        v2 = table[:, conditions[c]]
        stats[:, :, c] = v_tstat(v2, v1, alphas, axis=1)

        for j in range(len(tuning_param_range_values)):

            (u_hist, d_hist) = assign_bins(stats[j, :, c], num_bins, mins[j, c], maxes[j, c])
            d[j, c, :] = d_hist
            u[j, c, :] = u_hist

    return (u, d, stats)

    

def get_bins(n, maxval):

    # Bin 0 in the "up" histogram is for features that were down-regulated
    bins = []
    bins.extend(linspace(0, maxval, n))

    # Bin "numbin" in the "up" histogram is for features that were
    # above the max observed in the unpermuted data
    bins.append(inf)
    return bins


def make_bins(stats, maxes, mins, num_bins):
    """
    stats is an (l x m) array where l is the number of features and m
    is the number of tuning params.

    Maxes and mins are both (m x n) matrix where m is the number of
    tuning params and n is the number of conditions.

    Returns an (m x n x num_bins) array where m is the number of
    tuning params, n is the number of conditions, and num_bins is the
    number of bins.
    """
    pass
#    (l, m) = shape(stats)
    
#    (m2, n) = shape(maxes)

#    res = zeros((m, n, num_bins))

#    if m != m2: 
#        raise Exception("Ms aren't equal")




#    get_bins = [-inf, arange

#    for c in range(n): # Conditions
#        for i in range(l): # Features
#            for j in range(m): # Tuning params
#                val = stats[i, j]
#                if val >= center:
#                    bin_num = int(num_bins * (val - center) / (maxes[j, c] - center))
#                    u[j, c, bin_num] += 1
    

if __name__ == '__main__':
    main()
