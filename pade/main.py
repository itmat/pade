#!/usr/bin/env python

# Files:
#  + raw input file
#  + pade_schema.yaml
#  + pade_raw.hdf5
#  + pade_results.hdf5


"""The main program for pade."""

# External imports

import argparse
import collections
import errno
import jinja2
import logging
import numpy.ma as ma
import numpy as np
import h5py
from numpy.lib.recfunctions import append_fields
import os
import shutil
from itertools import combinations, product

from bisect import bisect

from pade.common import *
from pade.performance import *
from pade.schema import *
from pade.model import *
import pade.stat
from pade.stat import random_indexes, random_orderings, residuals, group_means, layout_is_paired
from pade.db import DB

REAL_PATH = os.path.realpath(__file__)
DEFAULT_TUNING_PARAMS=[0.001, 0.01, 0.1, 1, 3, 10, 30, 100, 300, 1000, 3000]

##############################################################################
###
### Exceptions
###

class UsageException(Exception):
    """Thrown when the user gave invalid parameters."""
    pass



def ensure_scores_increase(scores):
    """Returns a copy of the given ndarray with monotonically increasing values.

    """
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

    for grp in db.block_layout:
        means = np.mean(data[..., grp], axis=1)
        means = means.reshape(np.shape(means) + (1,))
        prediction[..., grp] = means
    return prediction

def summarize_by_conf_level(db):
    """Modify db to summarize the counts by conf level"""

    logging.info("Summarizing the results")

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
            param=db.tuning_params[db.best_param_idxs[i]])

    
def compute_coeffs(db):
    """Calculate the coefficients for the full model.

    :param db:
      The page.db.DB

    :returns:
      A (names, values) tuple where values is a (feature x coeff)
      array giving the value of each coefficient for each feature, and
      names is the list of names of the coefficients.

    """
    fitted = db.full_model.fit(db.table)
    names  = [assignment_name(a) for a in fitted.labels]    
    values = fitted.params
    return (names, values)


def compute_fold_change(db):
    """Compute fold change.

    :param db:
      The pade.db.DB

    :return:
      A (names, values) tuple where values is a (feature x group)
      array giving the fold change for each group for each feature,
      and names is the list of names for the groups.

    """
    logging.info("Computing fold change")
    
    full_factors     = set(db.full_model.expr.variables)
    nuisance_factors = set(db.reduced_model.expr.variables)
    test_factors     = full_factors.difference(nuisance_factors)

    if len(test_factors) > 1:
        raise UsageException(
            """Your full model must have exactly one more variable than the reduced model. We will change this soon.""")

    nuisance_assignments = db.schema.possible_assignments(nuisance_factors)
    fold_changes = []
    names = []

    data = db.table
    get_means = lambda a: np.mean(data[:, db.schema.indexes_with_assignments(a)], axis=-1)

    for na in nuisance_assignments:
        test_assignments = db.schema.possible_assignments(test_factors)
        test_assignments = [OrderedDict(d.items() + na.items()) for d in test_assignments]
        layouts = [ db.schema.indexes_with_assignments(a) for a in test_assignments ]
        baseline_mean = get_means(test_assignments[0])
        for a in test_assignments[1:]:
            fold_changes.append(get_means(a) / baseline_mean)
            names.append(assignment_name(a))

    # Ignoring nuisance vars
    test_assignments = db.schema.possible_assignments(test_factors)
    baseline_mean = get_means(test_assignments[0])
    for a in test_assignments[1:]:
        fold_changes.append(get_means(a) / baseline_mean)
        names.append(assignment_name(a))
        
    num_features = len(data)
    num_groups = len(names)

    result = np.zeros((num_features, num_groups))
    for i in range(len(fold_changes)):
        result[..., i] = fold_changes[i]

    return (names, result)


def compute_means(db):
    """Compute the means for each group in the full model.
    
    :param db:
      The pade.db.DB

    :return:
      A (names, values) tuple where values is a (feature x group)
      ndarray that gives the mean for each group for each feature. The
      groups are based on the 'full model' (combination of condition
      variables and blocking variables)
    
    """
    factors = db.full_model.expr.variables
    names = [assignment_name(a) 
             for a in db.schema.possible_assignments(factors)]
    values = get_group_means(db.schema, db.table, factors)
    return (names, values)
    
    
def do_run(args):
    print """
Analyzing {filename}, which is described by the schema {schema}.
""".format(filename=args.infile.name,
           schema=args.schema)
    db = args_to_db(args)
    import_table(db, args.infile.name)
    db.sample_indexes = new_sample_indexes(db)
    run_job(db, args.equalize_means_ids)

    (db.group_names, db.group_means) = compute_means(db)
    (db.coeff_names, db.coeff_values) = compute_coeffs(db)
    (db.fold_change_group_names, db.fold_change) = compute_fold_change(db)

    summarize_by_conf_level(db)
    print_summary(db)
    db.save()
    print """
The results for the job are saved in {path}. To generate a text
report, run:

  pade report --text --db {path}

To launch a small web server to generate the HTML reports, run:

  pade server --db {path}
""".format(path=db.path)

def do_server(args):
    import pade.server
    db = DB(path=args.db)
    db.load()
    pade.server.app.db = db
    if args.debug:
        pade.server.app.debug = True
    pade.server.app.run(port=args.port)
    

def do_report(args):

    db = DB(path=args.db)
    print """
Generating report for result database {db}.
""".format(db=db.path)

    db.load()

    makedirs(args.report_directory)
    with chdir(args.report_directory):
        if args.text:
            save_text_output(db)
            print "Saved text report to ", os.path.join(args.report_directory, "results.txt")
        #print_profile(db)

    print """
Reports are available in {loc}.
""".format(loc=args.report_directory)

def stat_for_name(db):
    """The statistic used for this job."""
    name = db.stat

    if db.is_paired:
        return pade.stat.OneSampleDifferenceTTest(
            block_layout=db.block_layout,
            condition_layout=db.condition_layout,
            alphas=db.tuning_params)

    elif name == 'f_test':

        if layout_is_paired(db.block_layout):
            raise UsageException(
"""I can't use the f-test with this data, because the reduced model
you specified has groups with only one sample. It seems like you have
a paired layout. If this is the case, please use the --paired option.
""")

        return pade.stat.Ftest(
            condition_layout=db.condition_layout,
            block_layout=db.block_layout,
            alphas=db.tuning_params)

    elif name == 'means_ratio':
        return pade.stat.MeansRatio(
            condition_layout=db.condition_layout,
            block_layout=db.block_layout,
            alphas=db.tuning_params)


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

    table = np.zeros((num_rows, num_cols), float)
    log_interval = int(num_rows / 10)
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

    # Tuning params
    if args.tuning_param is None or len(args.tuning_param) == 0 :
        db.tuning_params = np.array(DEFAULT_TUNING_PARAMS)
    else:
        db.tuning_params = np.array(args.tuning_param)

    # Num bins and samples
    db.num_bins = args.num_bins
    db.num_samples = args.num_samples

    logging.info("Creating db from args " + str(args))
    db.is_paired = args.paired

    # Stat name
    if db.is_paired:
        logging.info("You've given the --paired option, so I'll use a one-sample t-test.")
        stat_name = 'one_sample_t_test'
    else:
        stat_name = args.stat

    # Equalize means
    if stat_name in set(['one_sample_t_test', 'means_ratio']):
        logging.info("We're using stat " + stat_name + ", so I won't equalize means")
        equalize_means = False

    else:
        equalize_means = args.equalize_means

    db.stat = stat_name
    db.equalize_means = equalize_means
    
    blocks     = args.block     if args.block     is not None else []
    conditions = args.condition if args.condition is not None else []

    if len(blocks) > 0 or len(conditions) > 0:
        full_model = " * ".join(blocks + conditions)
        reduced_model = " * ".join(blocks)
        
    elif args.full_model is not None:
        full_model    = args.full_model
        reduced_model = args.reduced_model

    elif len(factors) == 1:
        full_model = db.schema.factors[0]
        reduced_model = None
    
    else:
        raise Exception("Since you have multiple factors, please specify a full model")

    db.full_model    = Model(db.schema, full_model)
    db.reduced_model = Model(db.schema, reduced_model)
    db.sample_from = args.sample_from
    db.sample_method = args.sample_method
    db.min_conf=args.min_conf
    db.summary_bins = np.arange(args.min_conf, 1.0, args.conf_interval)

    return db

@profiled
def run_job(db, equalize_means_ids):

    stat = stat_for_name(db)

    logging.info("Computing {stat} statistics on raw data".format(stat=stat.name))
    raw_stats = stat(db.table)
    logging.debug("Shape of raw stats is " + str(np.shape(raw_stats)))

    logging.info("Creating {num_bins} bins based on values of raw stats".format(
            num_bins=db.num_bins))
    db.bins = pade.stat.bins_uniform(db.num_bins, raw_stats)

    if db.sample_from == 'residuals':
        logging.info("Sampling from residuals")
        prediction = predicted_values(db)
        diffs      = db.table - prediction
        db.bin_to_mean_perm_count = pade.stat.bootstrap(
            prediction,
            stat, 
            indexes=db.sample_indexes,
            residuals=diffs,
            bins=db.bins)

    else:
        logging.info("Sampling from raw data")
        # Shift all values in the data by the means of the groups from
        # the full model, so that the mean of each group is 0.
        if db.equalize_means:
            shifted = residuals(db.table, db.full_layout)
            data = np.zeros_like(db.table)
            if equalize_means_ids is None:
                data = shifted
            else:
                ids = set([line.rstrip() for line in equalize_means_ids])
                count = len(ids)
                for i, fid in enumerate(db.feature_ids):
                    if fid in ids:
                        data[i] = shifted[i]
                        ids.remove(fid)
                    else:
                        data[i] = db.table[i]
                logging.info("Equalized means for " + str(count - len(ids)) + " features")
                if len(ids) > 0:
                    logging.warn("There were " + str(len(ids)) + " feature " +
                                 "ids given that don't exist in the data: " +
                                 str(ids))

            db.bin_to_mean_perm_count = pade.stat.bootstrap(
                data,
                stat, 
                indexes=db.sample_indexes,
                bins=db.bins)

        else:
            db.bin_to_mean_perm_count = pade.stat.bootstrap(
                db.table,
                stat, 
                indexes=db.sample_indexes,
                bins=db.bins)            

    logging.info("Done bootstrapping, now computing confidence scores")
    db.raw_stats    = raw_stats
    db.bin_to_unperm_count   = pade.stat.cumulative_hist(db.raw_stats, db.bins)
    db.bin_to_score = confidence_scores(
        db.bin_to_unperm_count, db.bin_to_mean_perm_count, np.shape(raw_stats)[-1])
    db.feature_to_score = assign_scores_to_features(
        db.raw_stats, db.bins, db.bin_to_score)

    return db


def save_text_output(db):

    (num_rows, num_cols) = db.table.shape
    num_cols += 2
    table = np.zeros((num_rows, num_cols))

    # Best tuning param for each feature
    idxs = np.argmax(db.feature_to_score, axis=0)
    table = []

    # For each row in the data, add feature id, stat, score, group
    # means, and raw values.
    logging.info("Building internal results table")
    for i in range(len(db.table)):
        row = []

        # Feature id
        row.append(db.feature_ids[i])
        
        # Best stat and all stats
        row.append(db.raw_stats[idxs[i], i])
        for j in range(len(db.tuning_params)):
            row.append(db.raw_stats[j, i])
        
        # Best score and all scores
        row.append(db.feature_to_score[idxs[i], i])
        for j in range(len(db.tuning_params)):
            row.append(db.feature_to_score[j, i])
        row.extend(db.group_means[i])
        row.extend(db.coeff_values[i])
        row.extend(db.table[i])
        table.append(tuple(row))
    schema = db.schema

    cols = []
    Col = namedtuple("Col", ['name', 'dtype', 'format'])
    def add_col(name, dtype, format):
        cols.append(Col(name, dtype, format))

    add_col(schema.feature_id_column_names[0], object, "%s")
    add_col('best_stat', float, "%f")
    for i, alpha in enumerate(db.tuning_params):
        add_col('stat_' + str(alpha), float, "%f")

    add_col('best_score', float, "%f")
    for i, alpha in enumerate(db.tuning_params):
        add_col('score_' + str(alpha), float, "%f")

    for name in db.group_names:
        add_col("mean: " + name, float, "%f")

    for name in db.coeff_names:
        add_col("param: " + name, float, "%f")

    for i, name in enumerate(schema.sample_column_names):
        add_col(name, float, "%f")
        
    dtype = [(c.name, c.dtype) for c in cols]

    logging.info("Writing table")
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

    # (unpermuted counts - mean permuted counts) / unpermuted counts
    res = (raw_counts - adjusted) / raw_counts

    for idx in np.ndindex(res.shape[:-1]):
        res[idx] = ensure_scores_increase(res[idx])

    return res


def get_group_means(schema, data, factors):
    logging.info("Getting group means for factors " + str(factors))
    assignments = schema.possible_assignments(factors=factors)

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

def new_sample_indexes(db):

    method  = (db.sample_method, db.sample_from)
    full    = db.full_model
    reduced = db.reduced_model
    R       = db.num_samples

    if method == ('perm', 'raw'):
        logging.info("Creating max of {0} random permutations".format(R))
        return list(random_orderings(db.condition_layout, db.block_layout, R))

    elif method == ('boot', 'raw'):
        logging.info("Bootstrapping raw values, within groups defined by '" + 
                     str(reduced.expr) + "'")
        return random_indexes(db.block_layout, R)
        
    elif method == ('boot', 'residuals'):
        logging.info("Bootstrapping using samples constructed from residuals, not using groups")
        return random_indexes(
            [ sorted(db.schema.sample_name_index.values()) ], R)

    # Paired design
    elif method == ('raw', 'diffs'):
        return list(random_orderings(db.condition_layout, db.block_layout, R))

    else:
        raise UsageException("Invalid sampling method")


def print_profile(db):

    walked = walk_profile()
    env = jinja2.Environment(loader=jinja2.PackageLoader('pade'))
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
    """Run pade."""

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
    elif args.func == do_server:
        console.setLevel(logging.INFO)
    else:
        console.setLevel(logging.ERROR)

    logging.getLogger('').addHandler(console)

    logging.info('Pade starting')

    try:
        args.func(args)
    except UsageException as e:
        logging.fatal("Pade exiting because of usage error")
        print fix_newlines(e.message)
        exit(1)
    
    logging.info('Pade finishing')    


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

Once you have finished the schema, you will need to run "pade run" to do the analysis. See "pade run -h" for its usage.
""").format(factors=schema.factors.keys(),
            filename=args.schema)

def add_reporting_args(p):
    grp = p.add_argument_group(
        title="reporting arguments")
    grp.add_argument(
        '--rows-per-pade',
        type=int,
        default=100,
        help="Number of rows to display on each pade of the report"),


def add_model_args(p):
    grp = p.add_argument_group(
        title="data model arguments",
        description="Use these options to specify the variables to use in the model.")
    
    grp.add_argument(
        '--block',
        '-b',
        help="""Specify a variable to use for blocking.""",
        action='append')
    
    grp.add_argument(
        '--condition',
        '-c',
        help="""Specify a variable that represents an experimental condition.""",
        action='append')

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
#        choices=['f', 't', 'f_sqrt'],
        choices=['f_test', 'one_sample_t_test', 'means_ratio'],
        default='f_test',
        help="The statistic to use. Only f-test is implemented at the moment, so this option has no effect.")

    grp.add_argument(
        '--tuning-param',
        type=float,
        action='append')

    grp.add_argument(
        '--paired', 
        action='store_true',
        default=False,
        help="Indicates that the input is paired")

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
        help="Number of bins to divide the statistic space into. You probably don't need to change this.")

    grp.add_argument(
        '--sample-method',
        default='perm',
        choices=['perm', 'boot'],
        help="""Whether to use a permutation test or bootstrapping to estimate confidence levels."""),

    grp.add_argument(
        '--min-conf',
        default=0.25,
        type=float,
        help="Smallest confidence level to report")

    grp.add_argument(
        '--conf-interval',
        default=0.05,
        type=float,
        help="Interval of confidence levels")

    grp.add_argument(
        '--no-equalize-means',
        action='store_false',
        dest='equalize_means',
        default=True,
        help="""Shift values of samples within same group for same feature so that their mean is 0 before the permutation test. This will likely cause Pade to be more conservative in selecting features.""")
    grp.add_argument(
        '--equalize-means-ids',
        type=file,
        help="""File giving list of feature ids to equalize means for. The file must contain each id by itself on its own line, with no header row.""")

def get_arguments():
    """Parse the command line options and return an argparse args
    object."""
    
    uberparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparsers = uberparser.add_subparsers(
        title='actions',
        description="""Normal usage is to run 'pade.py setup ...', then manually edit the
pade_schema.yaml file, then run 'pade.py run ...'.""")

    # Set up "parent" parser, which contains some arguments used by all other parsers
    parents = argparse.ArgumentParser(add_help=False)
    parents.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Be verbose")
    parents.add_argument(
        '--debug', '-d', 
        action='store_true',
        help="Print debugging information")
    parents.add_argument(
        '--log',
        default="pade.log",
        help="Location of log file")

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
        default="pade_schema.yaml")

    # Create "run" parser
    run_parser = subparsers.add_parser(
        'run',
        help="""Run the job.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[parents])
    run_parser.add_argument(
        '--schema', 
        help="The schema YAML file",
        default="pade_schema.yaml")
    run_parser.add_argument(
        'infile',
        help="""Name of input file""",
        default=argparse.SUPPRESS,
        type=file)
    run_parser.add_argument(
        '--db', 
        help="Path to the binary output file",
        default="pade_db.h5")
    
    add_model_args(run_parser)
    add_fdr_args(run_parser)

    report_parser = subparsers.add_parser(
        'report',
        help="""Generate report""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[parents])
    report_parser.add_argument(
        '--db', 
        help="Path to the db file to read results from",
        default="pade_db.h5")
    report_parser.add_argument(
        '--report-directory',
        default='pade_report',
        help="The directory to write the report to")
    report_parser.add_argument(
        '--text',
        action='store_true',
        help="Indicates that text reports should be produced")
    report_parser.add_argument(
        '--html',
        action='store_true',
        help="Indicates that HTML reports should be produced")

    add_reporting_args(report_parser)

    server_parser = subparsers.add_parser(
        'server',
        help="""Start server to show results""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[parents])
    server_parser.add_argument(
        '--db', 
        help="Path to the db file to read results from",
        default="pade_db.h5")
    server_parser.add_argument(
        '--port',
        type=int,
        help="Specify the port for the server to listen on")

    report_parser.set_defaults(func=do_report)
    run_parser.set_defaults(func=do_run)
    setup_parser.set_defaults(func=do_setup)
    server_parser.set_defaults(func=do_server)

    return uberparser.parse_args()


if __name__ == '__main__':
    main()
