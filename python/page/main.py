# External imports
import logging
import argparse
import os
import errno
import shutil
import numpy as np

from textwrap import fill

# PaGE imports
from schema import Schema
from common import Model, add_condition_axis
import report
import stats

class UsageException(Exception):
    pass


def fix_newlines(msg):
    output = ""
    for line in msg.splitlines():
        output += fill(line) + "\n"
    return output

class Job:

    DEFAULT_DIRECTORY = "pageseq_out"

    def __init__(self, 
                 directory, 
                 stat=None,
                 full_model=None,
                 reduced_model=Model()):
        self.directory = directory
        self.stat_name = stat
        self.full_model = full_model
        self.reduced_model = reduced_model
        self._table = None
        self._feature_ids = None

    @property
    def schema_filename(self):
        return os.path.join(self.directory, 'schema.yaml')

    @property
    def input_filename(self):
        return os.path.join(self.directory, 'input.txt')
    
    @property
    def schema(self):
        return Schema.load(open(self.schema_filename), self.input_filename)

    @property
    def stat(self):
        if self.stat_name == 'f':
            return stats.Ftest(
                layout_full=model_to_layout_map(self.schema, self.full_model).values(),
                layout_reduced=model_to_layout_map(self.schema, self.reduced_model).values())
        elif self.stat_name == 't':
            return stats.Ttest(alpha=1.0)

    @property
    def table(self):
        """Returns the data table as (sample x feature) ndarray."""
        if self._table is None:
            self._load_table()
        return self._table

    @property
    def feature_ids(self):
        if self._feature_ids is None:
            self._load_table()
        return self._feature_ids

    def _load_table(self):
        logging.info("Loading table from " + self.input_filename)
        with open(self.input_filename) as fh:

            headers = fh.next().rstrip().split("\t")

            ids = []
            table = []

            for line in fh:
                row = line.rstrip().split("\t")
                rowid = row[0]
                values = [float(x) for x in row[1:]]
                ids.append(rowid)
                table.append(values)

            self._table = np.array(table).swapaxes(0, 1)
            self._feature_ids = ids

            logging.info("Table shape is " + str(np.shape(self._table)))
            logging.info("Loaded " + str(len(self._feature_ids)) + " features")


    @property
    def num_features(self):
        return len(self.feature_ids)
            

def main():
    """Run pageseq."""

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='page.log',
                        filemode='w')

    args = get_arguments()

    console = logging.StreamHandler()
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
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
        print fix_newlines("ERROR: " + e.message)
    
    logging.info('Page finishing')    

def model_to_layout_map(schema, model):

    for factor in model.variables:
        if factor not in schema.factors:
            raise UsageException("Factor '" + factor + "' is not defined in the schema. Valid factors are " + str(schema.factors.keys()))
    
    return schema.sample_groups(model.variables)

def index_num_to_sym(schema, model, idx):
    res = tuple()
    for i, factor in enumerate(model.variables):
        values = list(schema.factor_values(factor))
        res += (factor, values[idx[i]])
    return res

def do_run(args):

    job = Job(
        directory=args.directory,
        stat=args.stat)

    full_model = args.full_model

    if full_model is None:
        if len(job.schema.factors) == 1:
            full_model = job.schema.factors[0]
        else:
            msg = """You need to specify a model with --full-model, since you have more than one factor ({0})."""
            raise UsageException(msg.format(job.schema.factors.keys()))

    schema = job.schema
    job.full_model = Model.parse(args.full_model)
    if args.reduced_model is not None:
        job.reduced_model = Model.parse(args.reduced_model)

    layout_map_full = model_to_layout_map(schema, job.full_model)
    logging.debug("Full layout map is " + str(layout_map_full))

    logging.info("Computing coefficients")
    coeffs = find_coefficients(job.schema, job.full_model, job.table)

    var_shape = schema.factor_value_shape(job.full_model.variables)
    layout = []
    group_keys = []
    names      = []
    (num_samples, num_features) = np.shape(job.table)
    num_groups = np.prod(var_shape)
    coeff_list = np.zeros((int(num_groups), num_features))

    for i, idx in enumerate(np.ndindex(var_shape)):
        key = index_num_to_sym(schema, job.full_model, idx)
        group_keys.append(key)
        layout.append(layout_map_full[key])
        key = index_num_to_sym(schema, job.full_model, idx)
        parts = []
        for j in range(0, len(key), 2):
            parts.append("{0}".format(key[j+1]))
        names.append("; ".join(parts))
        coeff_list[i] = coeffs[idx]

    logging.info("Computing means")
    reshaped = stats.apply_layout(layout, job.table)
    logging.debug("Shape of reshaped is " + str(np.shape(reshaped)))
    job.means = np.mean(reshaped, axis=0)
    logging.debug("Shape of means is " + str(np.shape(job.means)))

    logging.info("Computing statistics")
    job.stats = job.stat.compute(job.table)

    job.condition_names = names
    job.coeffs = coeff_list
    logging.debug("Condition names are" + str(job.condition_names))

    report.make_report(job)

    prediction = np.zeros_like(job.table)

    layout_map_red = model_to_layout_map(schema, job.reduced_model)

    for grp in layout_map_red.values():
        print grp
        data = job.table[grp]
        means = np.mean(data, axis=0)
        prediction[grp] = means

    bootstrap(job.table, prediction, job.stat, 1500)

def find_coefficients(schema, model, data):
    factors = model.variables

    layout_map = model_to_layout_map(schema, model)    
    values = [schema.sample_groups([f]).keys() for f in factors]
    shape = tuple([len(v) for v in values])
    (num_samples, num_features) = np.shape(data)
    shape += (num_features,)
    coeffs = np.zeros(shape)

    sym_idx    = lambda(idx): index_num_to_sym(schema, model, idx)
    col_nums   = lambda(idx): layout_map[sym_idx(idx)]
    group_mean = lambda(idx): np.mean(data[col_nums(idx)], axis=0)

    # Find the bias term
    bias_idx = tuple([0 for dim in shape[:-1]])
    coeffs[bias_idx] = group_mean(bias_idx)

    # Estimate the main effects
    for i, f in enumerate(factors):
        values = list(schema.factor_values(f))
        for j in range(1, len(values)):
            idx = np.zeros(len(shape) - 1, int)
            idx[i]= j
            idx = tuple(idx)
            coeffs[idx] = group_mean(idx) - coeffs[bias_idx]

    # Estimate the interactions between each pair of factors
    for i1 in range(len(factors)):
        f1 = factors[i1]
        vals1 = list(schema.factor_values(f1))

        for i2 in range(i1 + 1, len(factors)):
            f2 = factors[i2]
            vals2 = list(schema.factor_values(f2))
            for j1 in range(1, len(vals1)):

                idx1 = np.copy(bias_idx)
                idx1[i1] = j1
                idx1 = tuple(idx1)
                for j2 in range(1, len(vals2)):
                    idx2 = np.copy(bias_idx)
                    idx2[i2] = j2
                    idx2 = tuple(idx2)
                    idx = np.copy(bias_idx)
                    idx[i1] = j1
                    idx[i2] = j2
                    idx = tuple(idx)
                    coeffs[idx] = group_mean(idx) - coeffs[bias_idx] - coeffs[idx1] - coeffs[idx2]

    return coeffs

def bootstrap(data, prediction, stat_fn, R):
    
    # Number of samples
    m = np.shape(data)[0]

    # An R x m array where each row is a list of indexes into data,
    # randomly sampled.
    idxs = np.random.random_integers(0, m - 1, (R, m))

    # Find the error terms
    residuals = data - prediction
    
    # We'll return an R x n array, where n is the number of
    # features. Each row is the array of statistics for all the
    # features, using a different random sampling.
    results = np.zeros((R,) + np.shape(data)[1:])

    # Permute the error terms. For each permutation: Add the errors to
    # the data (or the predictions?) and compute the statistic again.
    for i, permutation in enumerate(idxs):
        permuted   = prediction + residuals[permutation]
        results[i] = stat_fn.compute(permuted)

    return results


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

    print fix_newlines("""
I am assuming that the input file is tab-delimited, with a header line. I am also assuming that the first column ({0}) contains feature identifiers, and that the rest of the columns ({1}) contain expression levels. In a future release, we will be more flexible about input formats. In the meantime, if this assumption is not true, you will need to reformat your input file.

""".format(headers[0],
           ", ".join(headers[1:])))

    return Schema(
        is_feature_id=is_feature_id,
        is_sample=is_sample,
        column_names=headers)

def init_job(infile, factors, directory, force=False):

    if isinstance(infile, str):
        infile = open(infile)
    schema = init_schema(infile=infile)    

    for a in factors:
        schema.add_factor(a, object)

    if not os.path.isdir(directory):
        os.makedirs(directory)
    job = Job(directory)

    mode = 'w' if force else 'wx'
    try:
        out = open(job.schema_filename, mode)
        schema.save(out)
    except IOError as e:
        if e.errno == errno.EEXIST:
            raise UsageException("The schema file \"{}\" already exists. If you want to overwrite it, use the --force or -f argument.".format(filename))
        raise e

    logging.info("Copying {0} to {1}".format(
        infile.name,
        job.input_filename))
    shutil.copyfile(infile.name, job.input_filename)
    return job

def do_setup(args):
    job = init_job(
        infile=args.infile,
        directory=args.directory,
        factors=args.factor,
        force=args.force)

    print fix_newlines("""
I have generated a schema for your input file, with factors {factors}, and saved it to "{filename}". You should now edit that file to set the factors for each sample. The file contains instructions on how to edit it.
""").format(factors=job.schema.factors.keys(),
            filename=job.schema_filename)


def get_arguments():
    """Parse the command line options and return an argparse args
    object."""
    
    uberparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    uberparser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Be verbose (print INFO level log messages)")

    uberparser.add_argument(
        '--debug', '-d', 
        action='store_true',
        help="Print debugging information")

    subparsers = uberparser.add_subparsers()

    # Setup
    setup_parser = subparsers.add_parser(
        'setup',
        description="""Set up the job configuration. This reads the input file and outputs a YAML file that you then need to fill out in order to properly configure the job.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
p        class.""")
    setup_parser.add_argument(
        '--directory', '-D',
        default=Job.DEFAULT_DIRECTORY,
        help="The directory to store the output data")
    setup_parser.add_argument(
        '--force', '-f',
        action='store_true',
        help="""Overwrite any existing files""")

    setup_parser.set_defaults(func=do_setup)

    # Run
    run_parser = subparsers.add_parser(
        'run',
        description="""Run the job.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    run_parser.add_argument(
        '--directory', '-d',
        default=Job.DEFAULT_DIRECTORY,
        help="""The directory to store the output data.

This directory must also contain schema.yaml file and input.txt""")
    run_parser.add_argument(
        '--full-model', '-M',
        help="""Specify the 'full' model. Required if there is more than one class.""")
    run_parser.add_argument(
        '--reduced-model', '-m',
        help="""Specify the 'reduced' model.""")

    run_parser.add_argument(
        '--stat', '-s',
        choices=['f', 't'],
        default='f',
        help="The statistic to use")
    run_parser.set_defaults(func=do_run)

    return uberparser.parse_args()


if __name__ == '__main__':
    main()
