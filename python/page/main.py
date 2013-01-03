# External imports

import argparse
import contextlib
import errno
import logging
import numpy as np
import os
import scipy.stats.mstats
import shutil
import tokenize

from StringIO import StringIO
from textwrap import fill
from jinja2 import Environment, PackageLoader

from schema import Schema
import stats

REAL_PATH = os.path.realpath(__file__)

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


##############################################################################
###
### Bare functions
###

def add_condition_axis(data, layout):

    num_samples = len(data)
    num_conditions = len(layout)
    num_samples_per_condition = len(layout[0])

    if num_conditions * num_samples_per_condition != num_samples:
        raise Exception("Bad layout")

    shape = (num_conditions, num_samples_per_condition) + np.shape(data)[1:]
    res = np.zeros(shape)

    for i, cond_cols in enumerate(layout):
        res[i] = data[layout[i]]

    return res


@contextlib.contextmanager
def chdir(path):
    cwd = os.getcwd()
    try:
        logging.info("Changing cwd from " + cwd + " to " + path)
        os.chdir(path)
        yield
    finally:
        logging.info("Changing cwd from " + path + " back to " + cwd)
        os.chdir(cwd)


def make_report(job):
    with chdir(job.directory):
        env = Environment(loader=PackageLoader('page'))
        setup_css(env)
        template = env.get_template('index.html')
        with open('index.html', 'w') as out:
            out.write(template.render(
                job=job
                ))

def setup_css(env):

    src = os.path.join(os.path.dirname(REAL_PATH),
                       '996grid/code/css')

    shutil.rmtree('css', True)
    shutil.copytree(src, 'css')

    with open('css/custom.css', 'w') as out:
        template = env.get_template('custom.css')
        out.write(template.render())

##############################################################################
###
### Classes
###


class Model:
    
    PROB_MARGINAL    = "marginal"
    PROB_JOINT       = "joint"

    OP_TO_NAME = { '+' : PROB_MARGINAL, '*' : PROB_JOINT }
    NAME_TO_OP = { PROB_MARGINAL : '+', PROB_JOINT : '*' }

    def __init__(self, prob=None, variables=[]):
        if prob is not None and prob not in self.NAME_TO_OP:
            raise Exception("Unknown probability " + str(prob))
        self.prob      = prob
        self.variables = variables
        
    @classmethod
    def parse(cls, string):
        """Parse a model from a string.

        string can either be "VARIABLE", "VARIABLE * VARIABLE", or
        "VARIABLE + VARIABLE". We may support more complex models
        later on.

        """
        operator = None
        variables = []

        io = StringIO(string)
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
            return Model(variables=variables)
        elif tok_type == tokenize.OP:
            operator = Model.OP_TO_NAME[tok]
            # raise ModelExpressionException("Unexpected operator " + tok)
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

        return Model(prob=operator, variables=variables)

    def __str__(self):
        if len(self.variables) == 1:
            return self.variables[0]
        elif len(self.variables) > 1:
            op = ' ' + self.NAME_TO_OP[self.prob] + ' '
            return op.join(self.variables)
        else:
            return ''

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

    make_report(job)

    prediction = np.zeros_like(job.table)

    layout_map_red = model_to_layout_map(schema, job.reduced_model)

    for grp in layout_map_red.values():
        print grp
        data = job.table[grp]
        means = np.mean(data, axis=0)
        prediction[grp] = means

    boot = Boot(job.table, prediction, job.stat, 1500)
    boot.ci()

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


class Boot:
    """Bootstrap results."""

    DEFAULT_LEVELS = np.linspace(0.5, 0.95, num=10)

    def __init__(self, data, prediction, stat_fn, R=1000):
        """Run bootstrapping.

        data - An m x ... array of data points

        prediction - An ndarray of the same shape as data, giving the
                     predicted values for the data points.

        stat_fn - A callable that takes data and returns a statistic
                  value.

        R - The number of samples to run. Defaults to 1000.
        
        """
        # Number of samples
        m = np.shape(data)[0]

        # An R x m array where each row is a list of indexes into data,
        # randomly sampled.
        idxs = np.random.random_integers(0, m - 1, (R, m))

        # We'll return an R x n array, where n is the number of
        # features. Each row is the array of statistics for all the
        # features, using a different random sampling.
        results = np.zeros((R,) + np.shape(data)[1:])

        # Permute the error terms. For each permutation: Add the errors to
        # the data (or the predictions?) and compute the statistic again.
        residuals = data - prediction
        for i, permutation in enumerate(idxs):
            permuted   = prediction + residuals[permutation]
            results[i] = stat_fn(permuted)

        self.permuted_stats = results
        self.raw_stats      = stat_fn(data)

    def ci(self, levels=DEFAULT_LEVELS):
        """Return the confidence intervals for the given levels.

        The result is an n x L array, where n is the number of
        features and L is the number of levels. ci[i, j] gives the
        value of the statistic for which the confidence of feature[i]
        being differentially expressed is at least levels[j].

        """
        intervals = scipy.stats.mstats.mquantiles(
            self.permuted_stats, prob=levels, axis=0)
        intervals = intervals.swapaxes(0, 1)
        print self.raw_stats[0]
        print sorted(self.permuted_stats[:, 0])
        print intervals[0]
        
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
