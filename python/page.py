import argparse
import re
import numpy as np
import itertools 
from numpy import *

from scipy.misc import comb

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
        self.table      = ma.masked_array(table, zeros(shape(table)))
#        self.table      = table
        self.column_indices = {}

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
    
def main():
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

    args = parser.parse_args()
    config = validate_args(args)
    data = load_input(config)
    dostuff(data)
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

    table = array(table)

    return Input(ids, headers[1:], table)

def unpermuted_means(data):
    num_conditions = data.num_conditions()
    num_features   = len(data.row_ids)

    res = zeros((num_features, num_conditions))

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

def find_default_alpha(data):

    baseline_cols = data.replicates(0)
    baseline_data = data.table[:,baseline_cols]

    alphas = zeros(data.num_conditions())

    for c in range(1, data.num_conditions()):
        cols = data.replicates(c)
        condition_data = data.table[:,cols]
        
        values = compute_s(condition_data, baseline_data, None, None, axis=1)

        the_mean = mean(values)

        lt_mean = values[values < the_mean]
        residuals = lt_mean - the_mean

        sd = sqrt(sum(residuals ** 2) / (len(residuals) - 1))
#        print "mean is %f, sd is %f, num is %d" % (the_mean, sd, len(lt_mean))

        alphas[c] = the_mean * 2 / sqrt(len(cols) + len(baseline_cols))

    return alphas

def dostuff(data):
    means = unpermuted_means(data)

tuning_param_range_values = [
    0.0001,
    0.01,
    0.1,
    0.3,
    0.5,
    1,
    1.5,
    2,
    3,
    10,
    ]


def v_tstat(v1, v2, tstat_tuning_param_default, axis=0):
    """
    Computes the t-statistic for two 2-D arrays. v1 is an m x n1 array
    and v2 is an m x n2 array, where m is the number of features, n1
    is the number of replicates in the condition represented by v1,
    and n2 is the number of conditions for v2. Returns an (m x p)
    array, where m again is the number of features, and p is the
    number of values in the tuning_param_range_values array.
    """

    sd1 = std(v1, ddof=1, axis=axis)
    sd2 = std(v2, ddof=1, axis=axis)

    len1 = np.array([len(row) for row in v1])
    len2 = np.array([len(row) for row in v2])

    S = sqrt((sd1**2*(len1-1) + sd2**2*(len2-1))/(len1 + len2 - 2))

    result = np.zeros((len(v1), len(tuning_param_range_values)))
    numer  = (mean(v1, axis=axis) - mean(v2, axis=axis)) * sqrt(len1 * len2)
    denom  = tstat_tuning_param_default * sqrt(len1 + len2)

    for i in range(0, len(tuning_param_range_values)):
        x = tuning_param_range_values[i]
        rhs = numer / ((x * tstat_tuning_param_default + S) * sqrt(len1 + len2))
        result[:,i] = rhs

    return result

def all_subsets(n, k):

    """
    Return a 2-D 
    """

    indexes = arange(n)
    m = comb(n, k)
    result = zeros((m, n))

    i = 0
    for subset in itertools.combinations(indexes, k):
        for j in subset:
            result[i,j] = 1
        i += 1
    return result

def min_max_stat(data, default_alphas):
    
    m = len(data.row_ids)
    n = data.num_conditions()

    table = zeros((m, len(tuning_param_range_values), n))

    for j in range(1, n):
        table[:,:,j] = v_tstat(data.table[:,data.replicates(j)],
                               data.table[:,data.replicates(0)],
                               default_alphas[j],
                               axis=1)

    mins  = np.min(table, axis=0)
    maxes = np.max(table, axis=0)
    
    return (mins, maxes)

if __name__ == 'main':
    main()
