#!/usr/bin/env python

# Files:
#  + raw input file
#  + page_schema.yaml
#  + page_raw.hdf5
#  + page_results.hdf5


"""The main program for page."""

# External imports

import argparse
import collections
import errno
import jinja2
import logging
import matplotlib.pyplot as plt
import numpy.ma as ma
import numpy as np
import h5py
from numpy.lib.recfunctions import append_fields
import os
import shutil
from itertools import combinations, product

from bisect import bisect

from page.common import *
from page.performance import *
from page.schema import *
from page.model import *
import page.stat
from page.stat import random_indexes, random_orderings, residuals, group_means
from page.db import DB

REAL_PATH = os.path.realpath(__file__)
RAW_VALUE_DTYPE = float
FEATURE_ID_DTYPE = 'S64'
DEFAULT_TUNING_PARAMS=np.array([0.001, 0.01, 0.1, 1, 3])

##############################################################################
###
### Exceptions
###

class UsageException(Exception):
    """Thrown when the user gave invalid parameters."""
    pass

##############################################################################
###
### Plotting and reporting
### 

StatDistPlot=namedtuple('StatDistPlot', ['tuning_param', 'filename'])

def plot_stat_dist(db, fdr):
    logging.info("Saving histograms of " + db.stat.name + " values")
    max_stat = np.max(fdr.raw_stats)
    for i, alpha in enumerate(DEFAULT_TUNING_PARAMS):
        filename = "images/raw_stats_" + str(i) + ".png"
        with figure(filename):
            plt.hist(fdr.raw_stats[i], log=False, bins=250)
            plt.title(db.stat.name + " distribution over features, $\\alpha = " + str(alpha) + "$")
            plt.xlabel(db.stat.name + " value")
            plt.ylabel("Features")
            plt.xlim(0, max_stat)
            yield StatDistPlot(alpha, filename)

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


def setup_css(env):
    """Copy the css from 996grid/code/css into its output location."""

    src = os.path.join(os.path.dirname(REAL_PATH), 'css')

    shutil.rmtree('css', True)
    shutil.copytree(src, 'css')

    with open('css/custom.css', 'w') as out:
        template = env.get_template('custom.css')
        out.write(template.render())


def ensure_scores_increase(scores):
    res = np.copy(scores)
    for i in range(1, len(res)):
        res[i] = max(res[i], res[i - 1])
    return res


@profiled
def predicted_values(db):
    """Return the values predicted by the reduced model.
    
    The return value has the same shape as the input table, with each
    cell containing the mean of all the cells in the same group, as
    defined by the reduced model.

    """
    data = db.table
    prediction = np.zeros_like(data)

    for grp in db.reduced_model.layout:
        means = np.mean(data[..., grp], axis=1)
        means = means.reshape(np.shape(means) + (1,))
        prediction[..., grp] = means
    return prediction

def summarize_by_conf_level(db):

    logging.info("Summarizing the results")

    db.summary_bins   = np.linspace(db.min_conf, 1.0, db.conf_levels)
    db.best_param_idxs = np.zeros(len(db.summary_bins))
    db.summary_counts = np.zeros(len(db.summary_bins))

    for i, conf in enumerate(db.summary_bins):
        idxs = db.feature_to_score > conf
        best = np.argmax(np.sum(idxs, axis=1))
        db.best_param_idxs[i] = best
        db.summary_counts[i]  = np.sum(idxs[best])

def print_summary(db):
    print """
Summary of features by confidence level:

Confidence |   Num.   | Tuning
   Level   | Features | Param.
-----------+----------+-------"""
    for i in range(len(db.summary_counts) - 1):
        print "{bin:10.1%} | {count:8d} | {param:0.4f}".format(
            bin=db.summary_bins[i],
            count=int(db.summary_counts[i]),
            param=DEFAULT_TUNING_PARAMS[db.best_param_idxs[i]])


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

def setup_sample_indexes(db):
    db.sample_indexes = new_sample_indexes(db)
           
def do_run(args):
    db = args_to_db(args)
    import_table(db, args.infile.name)
    setup_sample_indexes(db)
    run_job(db)
    summarize_by_conf_level(db)
    print_summary(db)
    db.save()

def do_report(args):

    db = DB(path=args.db)
    db.load()

    fitted = db.full_model.fit(db.table)
    means = get_group_means(db.schema, db.table)

    results = ResultTable(
        means=means,
        coeffs=fitted.params,
        group_names=[assignment_name(a) for a in db.schema.possible_assignments()],
        param_names=[assignment_name(a) for a in fitted.labels],
        feature_ids=np.array(db.feature_ids),
        stats=db.raw_stats,
        scores=db.feature_to_score)

    makedirs(args.report_directory)
    with chdir(args.report_directory):
        make_html_report(db, results, args.rows_per_page, args.report_directory)
        save_text_output(db, results)
        print_profile(db)

 
def stat_for_name(db):
    """The statistic used for this job."""
    name = db.stat
    if name == 'f':
        return page.stat.Ftest(
            layout_full=db.full_model.layout,
            layout_reduced=db.reduced_model.layout,
            alphas=DEFAULT_TUNING_PARAMS)
    elif name == 'f_sqrt':
        return page.stat.FtestSqrt(
            layout_full=db.full_model.layout,
            layout_reduced=db.reduced_model.layout)
    elif name == 't':
        return Ttest(alpha=1.0)


def import_table(db, path):
    logging.info("Loading table from " + path)
    logging.info("Counting rows and columns in input file")
    with open(path) as fh:

        headers = fh.next().rstrip().split("\t")
        num_cols = len(headers) - 1
        num_rows = 0
        for line in fh:
            num_rows += 1
        
    logging.info(
        "Input has {features} features and {samples} samples".format(
            features=num_rows,
            samples=num_cols))

    logging.info("Creating raw data table")

    table = np.zeros((num_rows, num_cols),
                     RAW_VALUE_DTYPE)
        
    log_interval=int(num_rows / 10)

    file = h5py.File(db.path, 'w')

    table = np.zeros((num_rows, num_cols))
    ids = []
        
    with open(path) as fh:

        headers = fh.next().rstrip().split("\t")

        for i, line in enumerate(fh):
            row = line.rstrip().split("\t")
            ids.append(row[0])
            table[i] = [float(x) for x in row[1:]]
            if (i % log_interval) == log_interval - 1:
                logging.debug("Copied {0} rows".format(i + 1))

    db.table = table
    db.feature_ids = ids

def args_to_db(args):
    db = DB(
        schema_path=args.schema,
        path=args.db)

    db.stat = args.stat
    db.num_bins = args.num_bins
    db.num_samples = args.num_samples
    db.full_model = Model(db.schema, args.full_model)
    db.reduced_model = Model(db.schema, args.reduced_model)
    db.sample_from = args.sample_from
    db.sample_method = args.sample_method
    db.min_conf=args.min_conf
    db.conf_levels=args.conf_levels

    return db

@profiled
def run_job(db):

    stat      = stat_for_name(db)

    logging.info("Computing {stat} statistics on raw data".format(stat=stat.name))
    raw_stats = stat(db.table)
    logging.debug("Shape of raw stats is " + str(np.shape(raw_stats)))

    logging.info("Creating {num_bins} bins based on values of raw stats".format(
            num_bins=db.num_bins))
    db.bins = page.stat.bins_uniform(db.num_bins, raw_stats)

    if db.sample_from == 'residuals':
        logging.info("Bootstrapping based on residuals")
        prediction = predicted_values(db)
        diffs      = db.table - prediction
        db.baseline_counts = page.stat.bootstrap(
            prediction,
            stat, 
            indexes=db.sample_indexes,
            residuals=diffs,
            bins=db.bins)
    else:
        logging.info("Bootstrapping based on raw data")
        # Shift all values in the data by the means of the groups from
        # the full model, so that the mean of each group is 0.
        shifted = residuals(db.table, db.full_model.layout)
        
        db.baseline_counts = page.stat.bootstrap(
            shifted,
            stat, 
            indexes=db.sample_indexes,
            bins=db.bins)

    logging.info("Done bootstrapping, now computing confidence scores")
    db.raw_stats    = raw_stats
    db.raw_counts   = page.stat.cumulative_hist(db.raw_stats, db.bins)
    db.bin_to_score = confidence_scores(
        db.raw_counts, db.baseline_counts, np.shape(raw_stats)[-1])
    db.feature_to_score = assign_scores_to_features(
        db.raw_stats, db.bins, db.bin_to_score)

    return db

def make_html_report(db, results, rows_per_page, directory):

    with profiling('do_report: build report'):

        extra = "\nstat " + db.stat_name + ", sampling " + db.sample_from
        env = jinja2.Environment(loader=jinja2.PackageLoader('page'))
        setup_css(env)

        template = env.get_template('conf_level.html')
        for level in range(len(db.summary_bins)):
            score=db.summary_bins[level]
            filtered = results.filter_by_score(score)
            pages = list(filtered.pages(rows_per_page))
            for page_num, page in enumerate(pages):
                with open('conf_level_{0}_page_{1}.html'.format(level, page_num), 'w') as out:
                    out.write(template.render(
                            conf_level=level,
                            min_score=score,
                            job=db,
                            results=page,
                            page_num=page_num,
                            num_pages=len(pages)))
                    

        template = env.get_template('index.html')
        with open('index.html', 'w') as out:
            out.write(template.render(
                    job=db,
                    results=results,
                    summary_bins=db.summary_bins,
                    summary_counts=db.summary_counts))

#        with chdir(job.directory):
#            stat_dist_template = env.get_template('stat_dist.html')
#            with open('html/stat_dist.html', 'w') as out:
#                out.write(stat_dist_template.render(
#                        plots=plot_stat_dist(job, fdr)))

    return (db, results)
    

def save_text_output(db, results):
    (num_rows, num_cols) = db.table.shape
    num_cols += 2
    table = np.zeros((num_rows, num_cols))

    # Best tuning param for each feature
    idxs = np.argmax(results.scores, axis=0)
    table = []

    # For each row in the data, add feature id, stat, score, group
    # means, and raw values.
    for i in range(len(db.table)):
        row = []

        # Feature id
        row.append(db.feature_ids[i])
        
        # Best stat and all stats
        row.append(db.raw_stats[idxs[i], i])
        for j in range(len(DEFAULT_TUNING_PARAMS)):
            row.append(db.raw_stats[j, i])
        
        # Best score and all scores
        row.append(db.feature_to_score[idxs[i], i])
        for j in range(len(DEFAULT_TUNING_PARAMS)):
            row.append(db.feature_to_score[j, i])
        row.extend(results.means[i])
        row.extend(results.coeffs[i])
        row.extend(db.table[i])
        table.append(tuple(row))
    schema = db.schema

    cols = []
    Col = namedtuple("Col", ['name', 'dtype', 'format'])
    def add_col(name, dtype, format):
        cols.append(Col(name, dtype, format))

    add_col(schema.feature_id_column_names[0], object, "%s")
    add_col('best_stat', float, "%f")
    for i, alpha in enumerate(DEFAULT_TUNING_PARAMS):
        add_col('stat_' + str(alpha), float, "%f")

    add_col('best_score', float, "%f")
    for i, alpha in enumerate(DEFAULT_TUNING_PARAMS):
        add_col('score_' + str(alpha), float, "%f")

    for name in results.group_names:
        add_col("mean: " + name, float, "%f")

    for name in results.param_names:
        add_col("param: " + name, float, "%f")

    for i, name in enumerate(schema.sample_column_names):
        add_col(name, float, "%f")
        
    dtype = [(c.name, c.dtype) for c in cols]


    table = np.array(table, dtype)
    with open("results.txt", "w") as out:
        out.write("\t".join(c.name for c in cols) + "\n")
        np.savetxt(out, table, 
                   fmt=[c.format for c in cols],
                   delimiter="\t")


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

    for idx in np.ndindex(res.shape[:-1]):
        res[idx] = ensure_scores_increase(res[idx])

    return res


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

def new_sample_indexes(self):

    method = (self.sample_method, self.sample_from)
        
    full = self.full_model
    reduced = self.reduced_model
    R = self.num_samples

    if method == ('perm', 'raw'):
        logging.info("Creating max of {0} random permutations".format(R))
        return list(random_orderings(full.layout, reduced.layout, R))

    elif method == ('boot', 'raw'):
        logging.info("Bootstrapping raw values, within groups defined by '" + 
                     str(reduced.expr) + "'")
        return random_indexes(reduced.layout, R)
        
    elif method == ('boot', 'residuals'):
        logging.info("Bootstrapping using samples constructed from residuals, not using groups")
        return random_indexes(
            [ sorted(self.schema.sample_name_index.values()) ], R)

    else:
        raise UsageException("Invalid sampling method")


def print_profile(db):

    walked = walk_profile()
    env = jinja2.Environment(loader=jinja2.PackageLoader('page'))
    setup_css(env)
    template = env.get_template('profile.html')
    with open('profile.html', 'w') as out:
        logging.info("Saving profile")
        out.write(template.render(profile=walked))

    fmt = []
    fmt += ["%d", "%d", "%s", "%f", "%f", "%f", "%f", "%f", "%f"]
    fmt += ["%d", "%d"]

    features = [ len(db.table) for row in walked ]
    samples  = [ len(db.table[0]) for row in walked ]

    walked = append_fields(walked, names='features', data=features)
    walked = append_fields(walked, names='samples', data=samples)


    with open('../profile.txt', 'w') as out:
        out.write("\t".join(walked.dtype.names) + "\n")
        np.savetxt(out, walked, fmt=fmt)


@profiled
def main():
    """Run pageseq."""

    args = get_arguments()

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=args.log,
                        filemode='w')

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


def init_schema(infile=None):
    """Creates a new schema based on the given infile.

    Does not save it or make any changes to the state of the file
    system.

    """
    if isinstance(infile, str):
        infile = open(infile)
    logging.info("Initializing schema from " + infile.name)
    header_line = infile.next().rstrip()    
    headers = header_line.split("\t")                
    is_feature_id = [i == 0 for i in range(len(headers))]
    is_sample     = [i != 0 for i in range(len(headers))]    

    return Schema(
        is_feature_id=is_feature_id,
        is_sample=is_sample,
        column_names=headers)


def init_job(infile, factors, schema_path=None, force=False):

    if isinstance(infile, str):
        infile = open(infile)
    schema = init_schema(infile=infile)    

    for name, values in factors.items():
        schema.add_factor(name, values)

    mode = 'w' if force else 'wx'
    try:
        with open(schema_path, mode) as out:
            logging.info("Saving schema to " + out.name)
            schema.save(out)
    except IOError as e:
        if e.errno == errno.EEXIST:
            raise UsageException("""\
                   The schema file \"{}\" already exists. If you want to
                   overwrite it, use the --force or -f argument.""".format(
                    schema_path))
        raise e

    return schema

@profiled
def do_setup(args):
    schema = init_job(
        infile=args.infile,
        schema_path=args.schema,
        factors={f : None for f in args.factor},
        force=args.force)

    print fix_newlines("""
I have generated a schema for your input file, with factors {factors}, and saved it to "{filename}". You should now edit that file to set the factors for each sample. The file contains instructions on how to edit it.
""").format(factors=schema.factors.keys(),
            filename=args.schema)

def add_reporting_args(p):
    grp = p.add_argument_group(
        title="reporting arguments")
    grp.add_argument(
        '--rows-per-page',
        type=int,
        default=100,
        help="Number of rows to display on each page of the report"),


def add_model_args(p):
    grp = p.add_argument_group(
        title="data model arguments",
        description="Use these options to specify the variables to use in the model.")
    
    grp.add_argument(
        '--full-model', '-M',

        help="""Specify the 'full' model. Required if there is more than one
class. For example, if you have factors 'batch' and 'treated', you could use
 'treated' or 'batch * treated'."""),

    grp.add_argument(
        '--reduced-model', '-m',
        help="""Specify the 'reduced' model. The format for the argument is the
 same as for --full-model."""),
    
    

def add_fdr_args(p):
    grp = p.add_argument_group(
        title="confidence estimation arguments",
        description="""These options control how we estimate the confidence levels. You can probably leave them unchanged, in which case I'll compute it using a permutation test with an f-test as the statistic, using a maximum of 1000 permutations.""")
    grp.add_argument(
        '--stat', '-s',
        choices=['f', 't', 'f_sqrt'],
        default='f',
        help="The statistic to use")

    grp.add_argument(
        '--num-samples', '-R',
        type=int,
        default=1000,
        help="The number of samples to use if bootstrapping, or the maximum number of permutations to use if doing permutation test.")

    grp.add_argument(
        '--sample-from',
        choices=['raw', 'residuals'],
        default='raw',
        help='Indicate whether to do bootstrapping based on samples from the raw data or sampled residuals')

    grp.add_argument(
        '--num-bins',
        type=int,
        default=1000,
        help="Number of bins to divide the statistic space into.")

    grp.add_argument(
        '--sample-method',
        default='perm',
        choices=['perm', 'boot'],
        help="""Whether to use a permutation test or bootstrapping to estimate confidence levels."""),

    grp.add_argument(
        '--min-conf',
        default=0.5,
        type=float,
        help="Smallest confidence level to report")

    grp.add_argument(
        '--conf-levels',
        default=11,
        type=int,
        help="Number of confidence levels to show")


def get_arguments():
    """Parse the command line options and return an argparse args
    object."""
    
    uberparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparsers = uberparser.add_subparsers(
        title='actions',
        description="""Normal usage is to run 'page.py setup ...', then manually edit the
page_schema.yaml file, then run 'page.py run ...'.""")

    # Set up "parent" parser, which contains some arguments used by all other parsers
    parents = argparse.ArgumentParser(add_help=False)
    parents.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Be verbose (print INFO level log messages)")
    parents.add_argument(
        '--debug', '-d', 
        action='store_true',
        help="Print debugging information")
    parents.add_argument(
        '--log',
        default="page.log",
        help="Location of log file")

    results_parents = argparse.ArgumentParser(add_help=False)
    results_parents.add_argument(
        '--results',
        help="PaGE results db",
        default="page_results.hdf5")

    # Create setup parser
    setup_parser = subparsers.add_parser(
        'setup',
        help="""Set up the job configuration. This reads the input file and
                outputs a YAML file that you then need to fill out in order to
                properly configure the job.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[parents])
    setup_parser.add_argument(
        'infile',
        help="""Name of input file""",
        default=argparse.SUPPRESS,
        type=file)
    setup_parser.add_argument(
        '--factor',
        action='append',
        required=True,
        help="""A class that can be set for each sample. You can
        specify this option more than once, to use more than one
        class.""")
    setup_parser.add_argument(
        '--force', '-f',
        action='store_true',
        help="""Overwrite any existing files""")
    setup_parser.add_argument(
        '--schema', 
        help="Path to write the schema file to",
        default="page_schema.yaml")

    setup_parser.set_defaults(func=do_setup)

    # Create "run" parser
    run_parser = subparsers.add_parser(
        'run',
        help="""Run the job.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[parents, results_parents])
    run_parser.add_argument(
        '--schema', 
        help="Path to read the schema file from",
        default="page_schema.yaml")
    run_parser.add_argument(
        'infile',
        help="""Name of input file""",
        default=argparse.SUPPRESS,
        type=file)
    run_parser.add_argument(
        '--db', 
        help="Path to the db file that I will create to store the results",
        default="page_db.h5")
    
    add_model_args(run_parser)
    add_fdr_args(run_parser)
    run_parser.set_defaults(func=do_run)

    report_parser = subparsers.add_parser(
        'report',
        help="""Generate report""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[parents, results_parents])
    report_parser.add_argument(
        '--db', 
        help="Path to the db file to read results from",
        default="page_db.h5")
    report_parser.add_argument(
        '--report-directory',
        default='page_report',
        help="The directory to write the report to")

    add_reporting_args(report_parser)
    report_parser.set_defaults(func=do_report)

    return uberparser.parse_args()


if __name__ == '__main__':
    main()
