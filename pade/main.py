#!/usr/bin/env python

# Files:
#  + raw input file
#  + pade_schema.yaml
#  + pade_raw.hdf5
#  + pade_results.hdf5


"""The main program for pade."""

# External imports

import argparse
import errno
import jinja2
import logging
import numpy as np
import h5py
from numpy.lib.recfunctions import append_fields
import os
import sys
import scipy.stats

from pade.common import *
from pade.performance import *
from pade.schema import *
from pade.model import *
import pade.stat
from pade.stat import random_indexes, random_orderings, residuals, group_means, layout_is_paired
from pade.job import Job
from pade.conf import *


REAL_PATH = os.path.realpath(__file__)
DEFAULT_TUNING_PARAMS=[0.001, 0.01, 0.1, 1, 3, 10, 30, 100, 300, 1000, 3000]

class UsageException(Exception):
    """Thrown when the user gave invalid parameters."""
    pass

@profiled
def predicted_values(job):
    """Return the values predicted by the reduced model.
    
    The return value has the same shape as the input table, with each
    cell containing the mean of all the cells in the same group, as
    defined by the reduced model.

    """
    data = job.input.table
    prediction = np.zeros_like(data)

    for grp in job.block_layout:
        means = np.mean(data[..., grp], axis=1)
        means = means.reshape(np.shape(means) + (1,))
        prediction[..., grp] = means
    return prediction

def summarize_by_conf_level(job):
    """Modify job to summarize the counts by conf level"""

    logging.info("Summarizing the results")

    job.results.summary_bins = np.arange(job.settings.min_conf, 1.0, job.settings.conf_interval)
    job.results.best_param_idxs = np.zeros(len(job.results.summary_bins))
    job.results.summary_counts = np.zeros(len(job.results.summary_bins))

    for i, conf in enumerate(job.results.summary_bins):
        idxs = job.results.feature_to_score > conf
        best = np.argmax(np.sum(idxs, axis=1))
        job.results.best_param_idxs[i] = best
        job.results.summary_counts[i]  = np.sum(idxs[best])

def print_summary(job):
    print """
Summary of features by confidence level:

Confidence |   Num.   | Tuning
   Level   | Features | Param.
-----------+----------+-------"""
    for i in range(len(job.results.summary_counts) - 1):
        print "{bin:10.1%} | {count:8d} | {param:0.4f}".format(
            bin=job.results.summary_bins[i],
            count=int(job.results.summary_counts[i]),
            param=job.settings.tuning_params[job.results.best_param_idxs[i]])

########################################################################
#
# Computing coefficients, fold change, and means
#
    
def compute_coeffs(job):
    """Calculate the coefficients for the full model.

    :param job:
      The page.job.Job

    :return: 
      A TableWithHeader giving the coefficients for the linear model
      for each feature.


    """
    fitted = job.full_model.fit(job.input.table)
    names  = [assignment_name(a) for a in fitted.labels]    
    values = fitted.params
    return pade.job.TableWithHeader(names, values)


def compute_fold_change(job):
    """Compute fold change.

    :param job:
      The pade.job.Job

    :return:
      A TableWithHeader giving the fold change for each non-baseline
      group for each feature.

    """
    logging.info("Computing fold change")
    
    nuisance_factors = set(job.settings.block_variables)
    test_factors     = job.settings.condition_variables

    if len(test_factors) > 1:
        raise UsageException(
            """You can only have one condition variable. We will change this soon.""")

    nuisance_assignments = job.schema.possible_assignments(nuisance_factors)
    fold_changes = []
    names = []

    data = job.input.table
    get_means = lambda a: np.mean(data[:, job.schema.indexes_with_assignments(a)], axis=-1)

    alpha = scipy.stats.scoreatpercentile(job.input.table.flatten(), 1.0)

    for na in nuisance_assignments:
        test_assignments = job.schema.possible_assignments(test_factors)
        test_assignments = [OrderedDict(d.items() + na.items()) for d in test_assignments]
        layouts = [ job.schema.indexes_with_assignments(a) for a in test_assignments ]
        baseline_mean = get_means(test_assignments[0])
        for a in test_assignments[1:]:
            fold_changes.append((get_means(a) + alpha) / (baseline_mean + alpha))
            names.append(assignment_name(a))

    # Ignoring nuisance vars
    test_assignments = job.schema.possible_assignments(test_factors)
    baseline_mean = get_means(test_assignments[0])
    for a in test_assignments[1:]:
        fold_changes.append((get_means(a) + alpha) / (baseline_mean + alpha))
        names.append(assignment_name(a))
        
    num_features = len(data)
    num_groups = len(names)

    result = np.zeros((num_features, num_groups))
    for i in range(len(fold_changes)):
        result[..., i] = fold_changes[i]

    return pade.job.TableWithHeader(names, result)


def compute_means(job):
    """Compute the means for each group in the full model.
    
    :param job:
      The pade.job.Job

    :return:
      A TableWithHeader giving the mean for each of the blocking and
      condition variables.
    
    """
    factors = job.settings.block_variables + job.settings.condition_variables
    names = [assignment_name(a) 
             for a in job.schema.possible_assignments(factors)]
    values = get_group_means(job.schema, job.input.table, factors)
    return pade.job.TableWithHeader(names, values)


########################################################################
#
# Handlers for command-line actions
#
    
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

def do_makesamples(args):

    settings=pade.job.Settings(
        sample_with_replacement=args.sample_with_replacement,
        num_samples=args.num_samples,
        block_variables=args.block,
        condition_variables=args.condition)

    job = Job(input=pade.job.Input.from_raw_file(args.infile.name, limit=1),
              settings=settings,
              schema=load_schema(args.schema))

    res = new_sample_indexes(job)
    if args.output is None:
        output = sys.stdout
    else:
        output = args.output
    np.savetxt(output, res, fmt='%d')

def load_sample_indexes(path):
    return np.genfromtxt(path, dtype=int)

def do_run(args):
    print """
Analyzing {filename}, which is described by the schema {schema}.
""".format(filename=args.infile.name,
           schema=args.schema)

    job = Job(input=pade.job.Input.from_raw_file(args.infile.name),
              settings=args_to_settings(args),
              schema=load_schema(args.schema),
              results=pade.job.Results())

    if args.sample_indexes is not None:
        logging.info("Loading sample indexes from user-specified file " + args.sample_indexes.name)
        job.results.sample_indexes = load_sample_indexes(args.sample_indexes)
    else:
        new_sample_indexes(job)

    run_job(job, args.equalize_means_ids)

    job.results.group_means  = compute_means(job)
    job.results.coeff_values = compute_coeffs(job)
    job.results.fold_change  = compute_fold_change(job)

    summarize_by_conf_level(job)
    print_summary(job)
    compute_orderings(job)

    pade.job.save_job(args.db, job)

    print """
The results for the job are saved in {path}. To generate a text
report, run:

  pade report --db {path}

To launch a small web server to generate the HTML reports, run:

  pade server --db {path}
""".format(path=args.db)

def do_server(args):
    import pade.server

    pade.server.app.job = pade.job.load_job(args.db)
    if args.debug:
        pade.server.app.debug = True
    pade.server.app.run(port=args.port)
    

def do_report(args):
    path = args.db

    print """
Generating report for result database {job}.
""".format(job=path)
    job = pade.job.load_job(path)
    filename = args.output
    save_text_output(job, filename=filename)
    print "Saved text report to ", filename


def get_stat_fn(job):
    """The statistic used for this job."""
    name = job.settings.stat_name

    if name == 'one_sample_t_test':
        constructor = pade.stat.OneSampleDifferenceTTest
    elif name == 'f_test':
        constructor = pade.stat.Ftest
    elif name == 'means_ratio':
        constructor = pade.stat.MeansRatio

    if constructor == pade.stat.Ftest and layout_is_paired(job.block_layout):
        raise UsageException(
"""I can't use the f-test with this data, because the reduced model
you specified has groups with only one sample. It seems like you have
a paired layout. If this is the case, please use the --paired option.
""")        

    return constructor(
        condition_layout=job.condition_layout,
        block_layout=job.block_layout,
        alphas=job.settings.tuning_params)


def args_to_settings(args):

    # If they said paired, override the choice of stat
    if args.paired:
        logging.info("You've given the --paired option, so I'll use a one-sample t-test.")
        stat = 'one_sample_t_test'
    else:
        stat = args.stat

    # If they chose one_sample_t_test or means_ratio, we can't
    # equalize the means.
    if stat in set(['one_sample_t_test', 'means_ratio']):
        logging.info("We're using stat " + stat + ", so I won't equalize means")
        equalize_means = False
    else:
        equalize_means = args.equalize_means

    # Block and condition variables
    if len(args.block) > 0 or len(args.condition) > 0:
        block_variables = args.block
        condition_variables = args.condition
        
    elif args.full_model is not None:
        full_model    = Model(schema, args.full_model)
        reduced_model = Model(schema, args.reduced_model)
        block_variables     = set(reduced_model.expr.variables)
        condition_variables = set(full_model.expr.variables).difference(block_vars)

    elif len(factors) == 1:
        condition_variables = factors
        block_variables = []
    
    else:
        raise Exception("Since you have multiple factors, please specify a full model")

    # Tuning params
    if args.tuning_param is None or len(args.tuning_param) == 0 :
        tuning_params = np.array(DEFAULT_TUNING_PARAMS)
    else:
        tuning_params = np.array(args.tuning_param)

    return pade.job.Settings(
        num_bins=args.num_bins,
        num_samples=args.num_samples,
        sample_from_residuals=args.sample_from_residuals,
        sample_with_replacement=args.sample_with_replacement,
        min_conf=args.min_conf,
        conf_interval=args.conf_interval,
        
        tuning_params=tuning_params,
        block_variables=block_variables,
        condition_variables=condition_variables,
        stat_name=stat,
        equalize_means=equalize_means
        )

def load_schema(path):
    try:
        with open(path) as f:
            return Schema.load(f)
    except IOError as e:
        raise UsageException("Couldn't load schema: " + e.filename + ": " + e.strerror)    

@profiled
def run_job(job, equalize_means_ids):

    stat = get_stat_fn(job)

    logging.info("Computing {stat} statistics on raw data".format(stat=stat.name))
    raw_stats = stat(job.input.table)
    logging.debug("Shape of raw stats is " + str(np.shape(raw_stats)))

    logging.info("Creating {num_bins} bins based on values of raw stats".format(
            num_bins=job.settings.num_bins))
    job.results.bins = pade.conf.bins_uniform(job.settings.num_bins, raw_stats)

    if job.settings.sample_from_residuals:
        logging.info("Sampling from residuals")
        prediction = predicted_values(job)
        diffs      = job.input.table - prediction
        job.results.bin_to_mean_perm_count = pade.conf.bootstrap(
            prediction,
            stat, 
            indexes=job.results.sample_indexes,
            residuals=diffs,
            bins=job.results.bins)

    else:
        logging.info("Sampling from raw data")
        # Shift all values in the data by the means of the groups from
        # the full model, so that the mean of each group is 0.
        if job.settings.equalize_means:
            shifted = residuals(job.input.table, job.full_layout)
            data = np.zeros_like(job.input.table)
            if equalize_means_ids is None:
                data = shifted
            else:
                ids = set([line.rstrip() for line in equalize_means_ids])
                count = len(ids)
                for i, fid in enumerate(job.input.feature_ids):
                    if fid in ids:
                        data[i] = shifted[i]
                        ids.remove(fid)
                    else:
                        data[i] = job.input.table[i]
                logging.info("Equalized means for " + str(count - len(ids)) + " features")
                if len(ids) > 0:
                    logging.warn("There were " + str(len(ids)) + " feature " +
                                 "ids given that don't exist in the data: " +
                                 str(ids))

            job.results.bin_to_mean_perm_count = pade.conf.bootstrap(
                data,
                stat, 
                indexes=job.results.sample_indexes,
                bins=job.results.bins)

        else:
            job.results.bin_to_mean_perm_count = pade.conf.bootstrap(
                job.input.table,
                stat, 
                indexes=job.results.sample_indexes,
                bins=job.results.bins)            

    logging.info("Done bootstrapping, now computing confidence scores")
    job.results.raw_stats    = raw_stats
    job.results.bin_to_unperm_count   = pade.conf.cumulative_hist(job.results.raw_stats, job.results.bins)
    job.results.bin_to_score = confidence_scores(
        job.results.bin_to_unperm_count, job.results.bin_to_mean_perm_count, np.shape(raw_stats)[-1])
    job.results.feature_to_score = assign_scores_to_features(
        job.results.raw_stats, job.results.bins, job.results.bin_to_score)

    return job


def save_text_output(job, filename):

    (num_rows, num_cols) = job.input.table.shape
    num_cols += 2
    table = np.zeros((num_rows, num_cols))

    # Best tuning param for each feature
    idxs = np.argmax(job.results.feature_to_score, axis=0)
    table = []

    # For each row in the data, add feature id, stat, score, group
    # means, and raw values.
    logging.info("Building internal results table")
    for i in range(len(job.input.table)):
        row = []

        # Feature id
        row.append(job.input.feature_ids[i])
        
        # Best stat and all stats
        row.append(job.results.raw_stats[idxs[i], i])
        for j in range(len(job.settings.tuning_params)):
            row.append(job.results.raw_stats[j, i])
        
        # Best score and all scores
        row.append(job.results.feature_to_score[idxs[i], i])
        for j in range(len(job.settings.tuning_params)):
            row.append(job.results.feature_to_score[j, i])
        row.extend(job.results.group_means[i])
        row.extend(job.results.coeff_values[i])
        row.extend(job.input.table[i])
        table.append(tuple(row))
    schema = job.schema

    cols = []
    Col = namedtuple("Col", ['name', 'dtype', 'format'])
    def add_col(name, dtype, format):
        cols.append(Col(name, dtype, format))

    add_col(schema.feature_id_column_names[0], object, "%s")
    add_col('best_stat', float, "%f")
    for i, alpha in enumerate(job.settings.tuning_params):
        add_col('stat_' + str(alpha), float, "%f")

    add_col('best_score', float, "%f")
    for i, alpha in enumerate(job.settings.tuning_params):
        add_col('score_' + str(alpha), float, "%f")

    for name in job.group_names:
        add_col("mean: " + name, float, "%f")

    for name in job.coeff_names:
        add_col("param: " + name, float, "%f")

    for i, name in enumerate(schema.sample_column_names):
        add_col(name, float, "%f")
        
    dtype = [(c.name, c.dtype) for c in cols]

    logging.info("Writing table")
    table = np.array(table, dtype)
    with open(filename, "w") as out:
        out.write("\t".join(c.name for c in cols) + "\n")
        np.savetxt(out, table, 
                   fmt=[c.format for c in cols],
                   delimiter="\t")


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


def new_sample_indexes(job):

    """Create array of sample indexes."""

    R  = job.settings.num_samples

    if job.settings.sample_with_replacement:
        if job.settings.sample_from_residuals:
            logging.info("Bootstrapping using samples constructed from " +
                         "residuals, not using groups")
            layout = [ sorted(job.schema.sample_name_index.values()) ]
        else:
            logging.info("Bootstrapping raw values, within groups defined by" + 
                         "'" + str(job.settings.block_variables) + "'")
            layout = job.block_layout
        logging.info("Layout is" + str(layout))
        return random_indexes(layout, R)

    else:
        logging.info("Creating max of {0} random permutations".format(R))
        return list(random_orderings(job.condition_layout, job.block_layout, R))

    
def print_profile(job):

    walked = walk_profile()
    env = jinja2.Environment(loader=jinja2.PackageLoader('pade'))
    template = env.get_template('profile.html')
    with open('profile.html', 'w') as out:
        logging.info("Saving profile")
        out.write(template.render(profile=walked))

    fmt = []
    fmt += ["%d", "%d", "%s", "%f", "%f", "%f", "%f", "%f", "%f"]
    fmt += ["%d", "%d"]

    features = [ len(job.input.table) for row in walked ]
    samples  = [ len(job.input.table[0]) for row in walked ]

    walked = append_fields(walked, names='features', data=features)
    walked = append_fields(walked, names='samples', data=samples)


    with open('../profile.txt', 'w') as out:
        out.write("\t".join(walked.dtype.names) + "\n")
        np.savetxt(out, walked, fmt=fmt)

def setup_logging(args):
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



@profiled
def main():
    """Run pade."""

    args = get_arguments()
    setup_logging(args)
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

########################################################################
#
# Parsing command line args
#

    
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

    model_parent = argparse.ArgumentParser(add_help=False)
    model_parent.add_argument(
        '--block',
        '-b',
        default=[],
        help="""Specify a variable to use for blocking. You can specify this multiple times if there are multiple blocking variables.""",
        action='append')
    
    model_parent.add_argument(
        '--condition',
        '-c',
        default=[],
        help="""Specify a variable that represents an experimental condition. Currently we only support one test condition.""",
        action='append')

    model_parent.add_argument(
        '--full-model', '-M',
        help=argparse.SUPPRESS) #"""Specify the 'full' model as an expression, like 'batch * treated'"""

    model_parent.add_argument(
        '--reduced-model', '-m',
        help=argparse.SUPPRESS) #"""Specify the 'reduced' model as an expression, like 'batch'."""

    # Sampling parent
    sampling_parent = argparse.ArgumentParser(add_help=False)
    sampling_parent.add_argument(
        '--num-samples', '-R',
        type=int,
        default=1000,
        help="The number of samples to use if bootstrapping, or the maximum number of permutations to use if doing permutation test.")
    sampling_parent.add_argument(
        '--sample-with-replacement',
        action='store_true',
        default=False,
        help="""Use sampling with replacement (bootstrapping) rather than permutation""")

    # Input file
    infile_parent = argparse.ArgumentParser(add_help=False)
    infile_parent.add_argument(
        'infile',
        help="""Name of input file""",
        default=argparse.SUPPRESS,
        type=file)
    schema_in_parent = argparse.ArgumentParser(add_help=False)
    schema_in_parent.add_argument(
        '--schema', 
        help="The schema YAML file to load",
        default="pade_schema.yaml")

    db_in_parent = argparse.ArgumentParser(add_help=False)
    db_in_parent.add_argument(
        '--db', 
        help="Path to the db file to read results from",
        default="pade_db.h5")


    ###
    ### Add sub-parsers
    ###

    setup_parser = subparsers.add_parser(
        'setup',
        help="""Set up the job configuration. This reads the input file and
                outputs a YAML file that you then need to fill out in order to
                properly configure the job.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[parents, infile_parent])

    run_parser = subparsers.add_parser(
        'run',
        help="""Run the job.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[parents, sampling_parent, infile_parent, schema_in_parent, model_parent])

    report_parser = subparsers.add_parser(
        'report',
        help="""Generate report""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[parents, db_in_parent])

    server_parser = subparsers.add_parser(
        'server',
        help="""Start server to show results""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[parents, db_in_parent])

    makesamples_parser = subparsers.add_parser(
        'makesamples',
        help="""Generate samples for a schema/input file combination""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[parents, infile_parent, sampling_parent, schema_in_parent, model_parent])

    ###
    ### Custom args for setup parser
    ###

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

    ###
    ### Custom args for run parser
    ###

    run_parser.add_argument(
        '--db', 
        help="Path to the binary output file",
        default="pade_db.h5")

    run_parser.add_argument(
        '--sample-indexes',
        type=file,
        help="""File specifying lists of indexes to use for sampling. See 'pade makesamples'.""")

    grp = run_parser.add_argument_group(
        title="confidence estimation arguments",
        description="""These options control how we estimate the confidence levels. You can probably leave them unchanged, in which case I'll compute it using a permutation test with an f-test as the statistic, using a maximum of 1000 permutations.""")
    grp.add_argument(
        '--stat', '-s',
#        choices=['f', 't', 'f_sqrt'],
        choices=['f_test', 'one_sample_t_test', 'means_ratio'],
        default='f_test',
        help="The statistic to use.")

    grp.add_argument(
        '--tuning-param',
        type=float,
        action='append',
        help="""For statistics that take some type of tuning parameter, you may optionally specify that parameter with this option. Specify it more than once and I'll search over all the tuning params and use the one that gives the highest power at each confidence level.""")

    grp.add_argument(
        '--paired', 
        action='store_true',
        default=False,
        help="Indicates that the input is paired. Synonym for '--stat one_sample_t_test'.")

    grp.add_argument(
        '--num-bins',
        type=int,
        default=1000,
        help="Number of bins to divide the statistic space into. You probably don't need to change this.")

    grp.add_argument(
        '--sample-from-residuals',
        default=False,
        action='store_true',
        help="""Sample from residuals rather than raw data.""")

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


    ###
    ### Custom args for report parser
    ###

    report_parser.add_argument(
        '--output', '-o',
        default="pade_report.txt",
        help="""Location to write report to""")

    ###
    ### Custom args for server parser
    ###

    server_parser.add_argument(
        '--port',
        type=int,
        help="Specify the port for the server to listen on")

    makesamples_parser.add_argument(
        '--output', '-o',
        help="File to write sample indexes to")

    report_parser.set_defaults(func=do_report)
    run_parser.set_defaults(func=do_run)
    setup_parser.set_defaults(func=do_setup)
    server_parser.set_defaults(func=do_server)
    makesamples_parser.set_defaults(func=do_makesamples)
    
    return uberparser.parse_args()

def compute_orderings(job):

    original = np.arange(len(job.input.feature_ids))
    stats = job.results.feature_to_score[...]
    rev_stats = 0.0 - stats

    by_score_original = np.zeros(np.shape(job.results.raw_stats), int)
    for i in range(len(job.settings.tuning_params)):
        by_score_original[i] = np.lexsort(
            (original, rev_stats[i]))

    job.results.order_by_score_original = by_score_original

    by_foldchange_original = np.zeros(np.shape(job.results.fold_change.table), int)
    foldchange = job.results.fold_change.table[...]
    rev_foldchange = 0.0 - foldchange
    for i in range(len(job.results.fold_change.header)):
        keys = (original, rev_foldchange[..., i])

        by_foldchange_original[..., i] = np.lexsort(keys)

    job.results.order_by_foldchange_original = by_foldchange_original



if __name__ == '__main__':
    main()
