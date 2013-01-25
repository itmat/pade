#!/usr/bin/env python

"""The main program for page."""

# External imports

import StringIO
import argparse
import collections
import errno
import jinja2
import logging
import matplotlib.pyplot as plt
import numbers
import numpy.ma as ma
import numpy as np
import os
import scipy.stats.mstats
import shutil
import textwrap
import tokenize
import numpy.lib.recfunctions
from itertools import combinations, product

from bisect import bisect

from page.common import *
from page.performance import *
from page.schema import *
import page.stat


from collections import OrderedDict

REAL_PATH = os.path.realpath(__file__)
RAW_VALUE_DTYPE = float
FEATURE_ID_DTYPE = 'S10'


##############################################################################
###
### Exceptions
###

class UsageException(Exception):
    """Thrown when the user gave invalid parameters."""
    pass

class ModelExpressionException(Exception):
    """Thrown when a model expression is invalid."""
    pass

class ProfileStackException(Exception):
    """Thrown when the profile stack is corrupt."""

##############################################################################
###
### Plotting and reporting
### 


def plot_conf_by_stat(fdr, filename='conf_by_stat',
                      extra=None):
    conf = fdr.bin_to_score
    bins = fdr.bins
    with figure(filename):
        plt.plot(bins[1:], conf, label='raw',)
        plt.plot(bins[1:], ensure_scores_increase(conf),
                 label='increasing')
        plt.xlabel('statistic')
        plt.ylabel('confidence')
        title = "Confidence level by statistic"
        if extra:
            title += extra
        plt.title(title)
        plt.semilogx()


def plot_counts_by_stat(fdr, filename='counts_by_stat', extra=None):
    raw_stats = fdr.raw_counts
    resampled_stats = fdr.baseline_counts
    bins = fdr.bins
    with figure(filename):
        plt.plot(bins[1:], raw_stats, label='raw data')
        plt.plot(bins[1:], resampled_stats, label='resampled')
        plt.loglog()
        title = 'Count of features by statistic'
        if extra is not None:
            title += extra
        plt.title(title)
        plt.xlabel('Statistic value')
        plt.ylabel('Features with statistic >= value')
        plt.legend()


def plot_raw_stat_hist(stats):
    """Plot the distribution of the given statistic values."""
    with figure("raw_stats"):
        plt.hist(stats, log=True, bins=250)


def setup_css(env):
    """Copy the css from 996grid/code/css into its output location."""

    src = os.path.join(os.path.dirname(REAL_PATH),
                       '../996grid/code/css')

    shutil.rmtree('css', True)
    shutil.copytree(src, 'css')

    with open('css/custom.css', 'w') as out:
        template = env.get_template('custom.css')
        out.write(template.render())

@profiled
def predicted_values(job):
    """Return the values predicted by the reduced model.
    
    The return value has the same shape as the input table, with each
    cell containing the mean of all the cells in the same group, as
    defined by the reduced model.
    """
    data = job.table
    prediction = np.zeros_like(data)

    for grp in job.reduced_model.layout:
        means = np.mean(data[..., grp], axis=1)
        means = means.reshape(np.shape(means) + (1,))
        prediction[..., grp] = means
    return prediction


def ensure_scores_increase(scores):
    res = np.copy(scores)
    for i in range(1, len(res)):
        res[i] = max(res[i], res[i - 1])
    return res


class FdrResults:
    """Simple data object used to back the HTML reports."""
    def __init__(self):
        self.bins = None
        self.raw_counts = None
        self.baseline_counts = None
        self.bin_to_score = None
        self.feature_to_to_score = None
        self.raw_stats = None
        self.summary_bins = None
        self.summary_counts = None

def summarize_scores(feature_to_score, summary_bins):

    shape = np.shape(feature_to_score)[:-1] + (len(summary_bins) - 1,)
    res = np.zeros(shape, int)
    for idx in np.ndindex(shape[:-1]):
        res[idx] = page.stat.cumulative_hist(feature_to_score[idx], summary_bins)
    return res

@profiled
def do_fdr(job):

    fdr = FdrResults()

    data = job.table
    stat = job.stat
    raw_stats = stat(data)

    fdr.bins = page.stat.bins_uniform(job.num_bins, raw_stats)

    with profiling('do_fdr.build fdr'):

        residuals = None

        if job.sample_from == 'raw':
            sample_layout = job.reduced_model.layout
            logging.info("Sampling from raw values, within groups defined by '" + 
                         str(job.reduced_model.expr) + "'")

        elif job.sample_from == 'residuals':
            logging.info("Sampling from residuals, not using groups")

            sample_layout = [ sorted(job.schema.sample_name_index.values()) ]
            prediction = predicted_values(job)
            residuals = data - prediction

        else:
            raise UsageException(
                "--sample-from must be either raw or residuals")

        logging.info("Setting up bootstrap sample indexes")
        if job.sample_indexes is None:
            logging.info("  Creating a new set of indexes")
            sample_indexes = page.stat.random_indexes(
                sample_layout, job.num_samples)
            job.save_sample_indexes(sample_indexes)
        else:
            logging.info("  Using existing indexes")

        fdr.baseline_counts = page.stat.bootstrap(
                data, stat, 
                indexes=job.sample_indexes,
                residuals=residuals,
                bins=fdr.bins)

        fdr.raw_stats    = raw_stats
        fdr.raw_counts   = page.stat.cumulative_hist(fdr.raw_stats, fdr.bins)
        fdr.bin_to_score = confidence_scores(
            fdr.raw_counts, fdr.baseline_counts, np.shape(raw_stats)[-1])
        fdr.feature_to_score = assign_scores_to_features(
            fdr.raw_stats, fdr.bins, fdr.bin_to_score)
        fdr.summary_bins = np.linspace(0.5, 1.0, 11)
        summarize_scores(fdr.feature_to_score, fdr.summary_bins)
        fdr.summary_counts = page.stat.cumulative_hist(
            fdr.feature_to_score, fdr.summary_bins)

    return fdr

class ResultTable:

    def __init__(self,
                 group_names=None,
                 param_names=None,
                 means=None,
                 coeffs=None,
                 stats=None,
                 feature_ids=None,
                 scores=None,
                 min_score=None):
        self.means = means
        self.coeffs = coeffs
        self.group_names = group_names
        self.param_names = param_names
        self.stats = stats
        self.feature_ids = feature_ids
        self.scores = scores
        self.min_score = min_score
        

    def filter_by_score(self, min_score):
        idxs = self.scores > min_score
        best = np.argmax(np.sum(idxs, axis=1))
        idxs = idxs[best]
        stats = self.stats[best]
        scores = self.scores[best]
        return ResultTable(
            group_names=self.group_names,
            param_names=self.param_names,
            means=self.means[idxs],
            coeffs=self.coeffs[idxs],
            stats=stats[idxs],
            feature_ids=self.feature_ids[idxs],
            scores=scores[idxs],
            min_score=min_score)

    def __len__(self):
        return len(self.feature_ids)

    def pages(self, rows_per_page=100):
        for start in range(0, len(self), rows_per_page):
            size = min(rows_per_page, len(self) - start)
            end = start + size

            yield ResultTable(
                group_names=self.group_names,
                param_names=self.param_names,
                means=self.means[start : end],
                coeffs=self.coeffs[start : end],
                stats=self.stats[start : end],
                feature_ids=self.feature_ids[start : end],
                scores=self.scores[start : end])



def assignment_name(a):

    if len(a) == 0:
        return "intercept"
    
    parts = ["{0}={1}".format(k, v) for k, v in a.items()]

    return ", ".join(parts)

            
@profiled
def do_run(args):

    with profiling("do_run: prologue"):

        job = Job(
            directory=args.directory,
            stat=args.stat,
            num_bins=args.num_bins,
            num_samples=args.num_samples,
            full_model=args.full_model,
            reduced_model=args.reduced_model,
            sample_from=args.sample_from)


        num_features = len(job.table)
        fitted = fit_model(job.full_model, job.table)
        means = get_group_means(job.schema, job.table)


    results = None

    fdr = do_fdr(job)

    with profiling("do_report: build results table"):

        results = ResultTable(
            means=means,
            coeffs=fitted.params,
            group_names=[assignment_name(a) for a in job.schema.possible_assignments()],
            param_names=[assignment_name(a) for a in fitted.labels],
            feature_ids=np.array(job.feature_ids),
            stats=fdr.raw_stats,
            scores=fdr.feature_to_score)

    with profiling('do_report: build report'):
        html_dir = os.path.join(job.directory, "html")
        makedirs(html_dir)
        with chdir(html_dir):
            extra = "\nstat " + job.stat_name + ", sampling " + job.sample_from
#           plot_counts_by_stat(fdr, extra=extra)
#           plot_conf_by_stat(fdr, extra=extra)
            env = jinja2.Environment(loader=jinja2.PackageLoader('page'))
            setup_css(env)

            summary_bins = fdr.summary_bins
            summary_counts = np.zeros(len(summary_bins))

            template = env.get_template('conf_level.html')
            for level in range(len(fdr.summary_bins)):
                score=fdr.summary_bins[level]
                filtered = results.filter_by_score(score)
                summary_counts[level] = len(filtered)
                pages = list(filtered.pages(args.rows_per_page))
                for page_num, page in enumerate(pages):
                    with open('conf_level_{0}_page_{1}.html'.format(level, page_num), 'w') as out:
                        out.write(template.render(
                                conf_level=level,
                                min_score=score,
                                job=job,
                                fdr=fdr,
                                results=page,
                                page_num=page_num,
                                num_pages=len(pages)))


            template = env.get_template('index.html')
            with open('index.html', 'w') as out:
                out.write(template.render(
                        job=job,
                        fdr=fdr,
                        results=results,
                        summary_bins=fdr.summary_bins,
                        summary_counts=summary_counts))

        print_profile()


    print """
Summary of features by confidence level:

Confidence |   Num.
   Level   | Features
-----------+---------"""
    for i in range(len(summary_counts) - 1):
        print "{bin:10.0%} | {count:8d}".format(
            bin=summary_bins[i],
            count=int(summary_counts[i]))

    print """
The full report is available at {0}""".format(
        os.path.join(job.directory, "index.html"))



@profiled
def assign_scores_to_features(stats, bins, scores):
    """Return an array that gives the confidence score for each feature.

    stats is an array giving the statistic value for each feature.

    bins is a monotonically increasing array which divides the
    statistic space up into ranges.

    scores is a monotonically increasing array of length (len(bins) -
    1) where scores[i] is the confidence level associated with
    statistics that fall in the range (bins[i-1], bins[i]).

    Returns an array that gives the confidence score for each feature.

    """
    logging.info("Assigning scores to features")
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

@profiled
def confidence_scores(raw_counts, perm_counts, num_features):
    """Return confidence scores.
    
    """
    logging.info("Getting confidence scores for shape {shape} with {num_features} features".format(shape=np.shape(raw_counts),
                                                                                                   num_features=num_features))
    if np.shape(raw_counts) != np.shape(perm_counts):
        raise Exception(
            """raw_counts and perm_counts must have same shape.
               raw_counts is {raw} and perm_counts is {perm}""".format(
                raw=np.shape(raw_counts), perm=np.shape(perm_counts)))
    
    shape = np.shape(raw_counts)
    adjusted = np.zeros(shape)
    for idx in np.ndindex(shape[:-1]):
        adjusted[idx] = adjust_num_diff(perm_counts[idx], raw_counts[idx], num_features)

    res = (raw_counts - adjusted) / raw_counts

    return res

def new_dummy_vars(schema, factors=None, level=None):
    """
    level=0 is intercept only
    level=1 is intercept plus main effects
    level=2 is intercept, main effects, interactions between two variables
    ...
    level=n is intercept, main effects, interactions between n variables

    """ 
    factors = schema._check_factors(factors)

    if level is None:
        return new_dummy_vars(schema, factors, len(factors))

    if level == 0:
        names = ({},)
        rows = []
        for a in schema.possible_assignments(factors):
            rows.append(
                DummyVarAssignment(a.values(), (True,), schema.samples_with_assignments(a)))
        return DummyVarTable(names, rows)

    res = new_dummy_vars(schema, factors, level - 1)

    col_names = tuple(res.names)
    rows      = list(res.rows)

    # Get col names
    for interacting in combinations(factors, level):
        for a in schema.factor_combinations(interacting):
            if schema.has_baseline(dict(zip(interacting, a))):
                continue
            col_names += ({ interacting[i] : a[i] for i in range(len(interacting)) },)

    for i, dummy in enumerate(rows):
        (factor_values, bits, indexes) = dummy

        # For each subset of factors of size level
        for interacting in combinations(factors, level):

            my_vals = ()
            for j in range(len(factors)):
                if factors[j] in interacting:
                    my_vals += (factor_values[j],)


            print "Look here\n"
            for a in schema.factor_combinations(interacting):
                print a

            for a in schema.possible_assignments(interacting):
                print a

            # For each possible assignment of values to these factors
            for a in schema.factor_combinations(interacting):
                if schema.has_baseline(dict(zip(interacting, a))):
                    continue

                # Test if this row of the result table has all the
                # values in this assignment
                matches = my_vals == a
                bits = bits + (matches,)

        rows[i] = DummyVarAssignment(tuple(factor_values), bits, indexes)

    return DummyVarTable(col_names, rows)




def fit_model(model, data):

    logging.info("Computing coefficients using least squares for " +
             str(len(data)) + " rows")

    effect_level = 1 if model.expr.operator == '+' else len(model.expr.variables)

    newvars = new_dummy_vars(model.schema, level=effect_level)

    x = []
    indexes = []

    for row in newvars.rows:
        for index in row.indexes:
            x.append(row.bits)
            indexes.append(model.schema.sample_name_index[index])

    x = np.array(x, bool)

    num_vars = np.size(x, axis=1)
    shape = np.array((len(data), num_vars))

    result = np.zeros(shape)

    print newvars
    for i, row in enumerate(data):
        y = row[indexes]
        (coeffs, residuals, rank, s) = np.linalg.lstsq(x, y)
        result[i] = coeffs

    return FittedModel(newvars.names, x, indexes, result)


def get_group_means(schema, data):

    assignments = schema.possible_assignments()

    num_features = len(data)
    num_groups = len(assignments)

    result = np.zeros((num_features, num_groups))

    for i, assignment in enumerate(assignments):
        indexes = schema.indexes_with_assignments(assignment)
        result[:, i] = np.mean(data[:, indexes], axis=1)

    return result


##############################################################################
###
### Classes
###


class ModelExpression:
    
    def __init__(self, 
                 operator=None,
                 variables=None):

        if variables is None:
            variables = []

        # If we have two or mor vars, make sure we have a valid operator
        if len(variables) > 1:
            if operator is None:
                raise Exception("Operator must be supplied for two or more variables")
            if operator not in "+*":
                raise Exception("Operator must be '+' or '*'")

        self.operator = operator
        self.variables = variables

        
    @classmethod
    def parse(cls, string):
        """Parse a model from a string.

        string can either be "VARIABLE", "VARIABLE * VARIABLE", or
        "VARIABLE + VARIABLE". We may support more complex models
        later on.

        """

        if string is None or string.strip() == '':
            return ModelExpression()

        operator = None
        variables = []

        io = StringIO.StringIO(string)
        toks = tokenize.generate_tokens(lambda : io.readline())

        # First token should always be a variable
        t = toks.next()
        (tok_type, tok, start, end, line) = t
        if tok_type != tokenize.NAME:
            raise ModelExpressionException("Unexpected token " + tok)
        variables.append(tok)

        # Second token should be either the end marker or + or *
        t = toks.next()
        (tok_type, tok, start, end, line) = t
        if tok_type == tokenize.ENDMARKER:
            return ModelExpression(variables=variables)
        elif tok_type == tokenize.OP:
            operator = tok
        else:
            raise ModelExpressionException("Unexpected token " + tok)

        t = toks.next()
        (tok_type, tok, start, end, line) = t
        if tok_type != tokenize.NAME:
            raise ModelExpressionException("Unexpected token " + tok)
        variables.append(tok)

        t = toks.next()
        (tok_type, tok, start, end, line) = t
        if tok_type != tokenize.ENDMARKER:
            raise ModelExpressionException("Expected end of expression, got " + tok)

        if len(variables) > 0:
            return ModelExpression(operator=operator, variables=variables)



    def __str__(self):
        if len(self.variables) == 0:
            return ""
        elif len(self.variables) == 1:
            return self.variables[0]
        else:
            op = " " + self.operator + " "
            return op.join(self.variables)


class Model:
    def __init__(self, schema, expr):
        self.schema = schema
        self.expr = ModelExpression.parse(expr)
        self.validate_model()

    def validate_model(self):
        """Validate the model against the given schema.

        Raises an exception if the model refers to any variables that are
        not defined in schema.
        
        """
        
        for factor in self.expr.variables:
            if factor not in self.schema.factors:
                raise UsageException("Factor '" + factor + "' is not defined in the schema. Valid factors are " + str(self.schema.factors.keys()))

    @property
    def layout(self):
        s = self.schema
        return [s.indexes_with_assignments(a)
                for a in s.possible_assignments(self.expr.variables)]
    
    def index_num_to_sym(self, idx):
        schema = self.schema
        expr  = self.expr
        res = tuple()
        for i, factor in enumerate(expr.variables):
            values = list(schema.factor_values(factor))
            res += (values[idx[i]],)
        return res


class Job:

    DEFAULT_DIRECTORY = "pageseq_out"

    def __init__(self, 
                 directory, 
                 stat=None,
                 full_model=None,
                 reduced_model=None,
                 num_bins=None,
                 num_samples=None,
                 sample_from=None,
                 schema=None
                 ):
        self.directory = directory
        self.stat_name = stat
        self._table = None
        self._feature_ids = None

        # FDR configuration
        self.num_bins = num_bins
        self.num_samples = num_samples
        self.sample_from = sample_from

        if schema is None:
            self.schema = Schema.load(open(self.schema_path))
        else:
            self.schema = schema

        self.full_model    = Model(self.schema, full_model)
        self.reduced_model = Model(self.schema, reduced_model)

        self._sample_indexes = None

    def save_sample_indexes(self, indexes):
        self._sample_indexes = indexes
        np.save(self.sample_indexes_path, indexes)

    @property
    def sample_indexes(self):
        logging.debug("In Job.sample_indexes")
        if self._sample_indexes is None:
            path = self.sample_indexes_path

            logging.debug("_sample_indexes is none; initializing. Looking at " + 
                          path)
            if os.path.exists(path):
                logging.debug("Found sample indexes to load")
                self._sample_indexes = np.load(path)
            else:
                logging.info("Sample indexes aren't created yet")
        return self._sample_indexes

    def save_schema(self, mode):
        out = open(self.schema_path, mode)
        self.schema.save(out)

    @property
    def num_features(self):
        return len(self.feature_ids)

    @property
    def data_directory(self):
        return os.path.join(self.directory, 'data')

    @property
    def table_path(self):
        return os.path.join(self.data_directory, 'table.npy')

    @property
    def feature_ids_path(self):
        return os.path.join(self.data_directory, 'feature_ids.npy')

    @property
    def sample_indexes_path(self):
        """Path to table of sample indexes"""
        return os.path.join(self.data_directory, 'sample_indexes.npy')

    @property
    def schema_path(self):
        """The path to the schema."""
        return os.path.join(self.directory, 'schema.yaml')

    @property
    def input_path(self):
        """Path to the input file."""
        return os.path.join(self.directory, 'input.txt')
    
    @property
    def stat(self):
        """The statistic used for this job."""
        if self.stat_name == 'f':
            return page.stat.Ftest(
                layout_full=self.full_model.layout,
                layout_reduced=self.reduced_model.layout,
                alphas=np.array([0.0, 0.01, 0.1, 1, 3]))
        elif self.stat_name == 'f_sqrt':
            return page.stat.FtestSqrt(
                layout_full=self.full_model.layout,
                layout_reduced=self.reduced_model.layout)
        elif self.stat_name == 't':
            return Ttest(alpha=1.0)

    @property
    def table(self):
        """The data table as a (sample x feature) ndarray."""
        if self._table is None:
            shape = (self.num_features,
                     len(self.schema.sample_column_names))
            self._table = np.memmap(
                self.table_path, 
                mode='r',
                shape=shape,
                dtype=RAW_VALUE_DTYPE)
        return self._table

    @property
    def swapped_table(self):
        return self._table.swapaxes(0, )

    @property
    def feature_ids(self):
        """Array of the ids of features from my input file."""
        if self._feature_ids is None:
            self._feature_ids = np.memmap(
                self.feature_ids_path,
                mode='r',
                dtype=FEATURE_ID_DTYPE)

        return self._feature_ids

    def copy_table(self, raw_input_path):
        logging.info("Loading table from " + raw_input_path +
                     " to " + self.table_path + " and " + self.feature_ids_path)
        
        logging.info("Counting rows and columns in input file")
        with open(raw_input_path) as fh:

            headers = fh.next().rstrip().split("\t")
            num_cols = len(headers) - 1
            num_rows = 0
            for line in fh:
                num_rows += 1
        
        logging.info(
            "Input has {features} features and {samples} samples".format(
                features=num_rows,
                samples=num_cols))

        logging.info(
            "Creating raw data table")

        table = np.memmap(
            self.table_path, 
            mode='w+',
            dtype=RAW_VALUE_DTYPE,
            shape=(num_rows, num_cols))

        ids = np.memmap(
            self.feature_ids_path,
            mode='w+',
            dtype=FEATURE_ID_DTYPE,
            shape=(num_rows,))

        log_interval=int(num_rows / 10)

        with open(raw_input_path) as fh:

            headers = fh.next().rstrip().split("\t")

            for i, line in enumerate(fh):
                row = line.rstrip().split("\t")
                ids[i]   = row[0]
                table[i] = [float(x) for x in row[1:]]
                if (i % log_interval) == log_interval - 1:
                    logging.debug("Copied {0} rows".format(i + 1))


        del table
        del ids

def print_profile():

    walked = walk_profile()
    env = jinja2.Environment(loader=jinja2.PackageLoader('page'))
    setup_css(env)
    template = env.get_template('profile.html')
    with open('profile.html', 'w') as out:
        out.write(template.render(profile=walked))

def main():
    """Run pageseq."""

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename='page.log',
                        filemode='w')

    args = get_arguments()

    console = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)-8s %(message)s')
    console.setFormatter(formatter)

    if args.debug:
        console.setLevel(logging.DEBUG)
    elif args.verbose:
        console.setLevel(logging.INFO)
    else:
        console.setLevel(logging.ERROR)

    logging.getLogger('').addHandler(console)

    logging.info('Page starting')

    try:
        args.func(args)
    except UsageException as e:
        logging.fatal("Page exiting because of usage error")
        print fix_newlines(e.message)
        exit(1)
    
    logging.info('Page finishing')    

# Factor = collections.namedtuple("Factor", ["name", "dtype", "values"])
# Sample = collections.namedtuple("Sample", ["column", "factor_values"])

DummyVarTable = collections.namedtuple(
    "DummyVarTable",
    ["names", "rows"])

DummyVarAssignment = collections.namedtuple(
    "DummyVarAssignment",
    ["factor_values",
     "bits",
     "indexes"])

FittedModel = collections.namedtuple(
    "FittedModel",
    ["labels",
     "x",
     "y_indexes",
     "params"])



def init_schema(infile=None):
    """Creates a new schema based on the given infile.

    Does not save it or make any changes to the state of the file
    system.

    """
    if isinstance(infile, str):
        infile = open(infile)
    header_line = infile.next().rstrip()    
    headers = header_line.split("\t")                
    is_feature_id = [i == 0 for i in range(len(headers))]
    is_sample     = [i != 0 for i in range(len(headers))]    

    return Schema(
        is_feature_id=is_feature_id,
        is_sample=is_sample,
        column_names=headers)

def init_job(infile, factors, directory, force=False):

    if isinstance(infile, str):
        infile = open(infile)
    schema = init_schema(infile=infile)    

    for name, values in factors.items():
        schema.add_factor(name, values)

    job = Job(directory, schema=schema)
    makedirs(job.data_directory)

    mode = 'w' if force else 'wx'
    try:
        job.save_schema(mode)
    except IOError as e:
        if e.errno == errno.EEXIST:
            raise UsageException("""\
                   The schema file \"{}\" already exists. If you want to
                   overwrite it, use the --force or -f argument.""".format(
                    job.schema_path))
        raise e

    job.copy_table(infile.name)

    return job

@profiled
def do_setup(args):
    job = init_job(
        infile=args.infile,
        directory=args.directory,
        factors={f : None for f in args.factor},
        force=args.force)

    print fix_newlines("""
I have generated a schema for your input file, with factors {factors}, and saved it to "{filename}". You should now edit that file to set the factors for each sample. The file contains instructions on how to edit it.
""").format(factors=job.schema.factors.keys(),
            filename=job.schema_path)

ARGUMENTS = {
    'infile' : lambda p: p.add_argument(
        'infile',
        help="""Name of input file""",
        default=argparse.SUPPRESS,
        type=file),

    'factor' : lambda p: p.add_argument(
        '--factor',
        action='append',
        required=True,
        help="""A class that can be set for each sample. You can
        specify this option more than once, to use more than one
p        class."""),

    'verbose' : lambda p: p.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Be verbose (print INFO level log messages)"),
    
    'debug' : lambda p: p.add_argument(
        '--debug', '-d', 
        action='store_true',
        help="Print debugging information"),

    'directory' : lambda p: p.add_argument(
        '--directory', '-D',
        default=Job.DEFAULT_DIRECTORY,
        help="The directory to store the output data"),

    'force' : lambda p: p.add_argument(
        '--force', '-f',
        action='store_true',
        help="""Overwrite any existing files"""),
    
    'full_model' : lambda p: p.add_argument(
        '--full-model', '-M',

        help="""Specify the 'full' model. Required if there is more than one 
class. For example, if you have factors 'batch' and 'treated', you could use
 'treated' or 'batch * treated'."""),

    'reduced_model' : lambda p: p.add_argument(
        '--reduced-model', '-m',
        help="""Specify the 'reduced' model. The format for the argument is the
 same as for --full-model."""),
    
    'stat' : lambda p: p.add_argument(
        '--stat', '-s',
        choices=['f', 't', 'f_sqrt'],
        default='f',
        help="The statistic to use"),
    
    'num_samples' : lambda p: p.add_argument(
        '--num-samples', '-R',
        type=int,
        default=1000,
        help="The number of samples to use for bootstrapping"),

    'sample_from' : lambda p: p.add_argument(
        '--sample-from',
        choices=['raw', 'residuals'],
        default='residuals',
        help='Indicate whether to do bootstrapping based on samples from the raw data or sampled residuals'),

    'num_bins' : lambda p: p.add_argument(
        '--num-bins',
        type=int,
        default=1000,
        help="Number of bins to divide the statistic space into."),

    'rows_per_page' : lambda p: p.add_argument(
        '--rows-per-page',
        type=int,
        default=100,
        help="Number of rows to display on each page of the report")

}

def add_args(parser, args):
    for a in args:
        ARGUMENTS[a](parser)


def get_arguments():
    """Parse the command line options and return an argparse args
    object."""
    
    uberparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparsers = uberparser.add_subparsers(
        title='actions',
        description="""Normal usage is to run 'page.py setup ...', then manually edit the
schema.yaml file, then run 'page.py run ...'.""")

    # Setup
    setup_parser = subparsers.add_parser(
        'setup',
        help="""Set up the job configuration. This reads the input file and
                outputs a YAML file that you then need to fill out in order to
                properly configure the job.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    add_args(setup_parser, ['infile', 'factor', 'directory', 'verbose', 'debug', 'force'])

    setup_parser.set_defaults(func=do_setup)

    # Prepare
    prep_parser = subparsers.add_parser(
        'prep',
        help="""Prepare the input file for analysis, generating the samples 
                for bootstrapping and splitting the input file (if --chunks
                is supplied).""")

    add_args(prep_parser, [
            'directory', 'full_model', 'reduced_model', 'num_samples',
            'verbose', 'debug'])

    # Run
    run_parser = subparsers.add_parser(
        'run',
        help="""Run the job.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    add_args(run_parser, [
            'directory', 'full_model', 'reduced_model', 'num_samples', 
            'num_bins', 'rows_per_page', 'stat', 'verbose', 'debug', 
            'sample_from'])

    run_parser.set_defaults(func=do_run)

    return uberparser.parse_args()


if __name__ == '__main__':
    main()
