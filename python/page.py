# External imports

import StringIO
import argparse
import collections
import contextlib
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
import yaml


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

def double_sum(data):
    """Returns the sum of data over the first two axes."""
    return np.sum(np.sum(data, axis=0), axis=0)

def apply_layout(layout, data):
    """Reshape the given data so based on the layout.

    layout - A list of lists. Each inner list is a sequence of indexes
    into data, representing a group of samples that share the same
    factor values. All of the inner lists must currently have the same
    length.

    data - An m x n array, where m is the number of samples and n is
    the number of features.

    Returns an m1 x m2 x n array, where m1 is the number of groups, m2
    is the number of samples in each group, and n is the number of features.

    """
    shape = (len(layout), len(layout[0])) + np.shape(data)[1:]

    res = np.zeros(shape)
    for i, idxs in enumerate(layout):
        res[i] = data[idxs]
    return res.swapaxes(0, 1)


def fix_newlines(msg):
    """Attempt to wrap long lines as paragraphs.

    Treats each line in the given message as a paragraph. Wraps each
    paragraph to avoid long lines. Returns a string that contains all
    the wrapped paragraphs, separated by blank lines.

    """
    output = ""
    for line in msg.splitlines():
        output += textwrap.fill(line) + "\n"
    return output


def make_report(job):
    """Make the HTML report for the given job."""

    with chdir(job.directory):
        env = jinja2.Environment(loader=jinja2.PackageLoader('page'))
        setup_css(env)
        template = env.get_template('index.html')
        with open('index.html', 'w') as out:
            out.write(template.render(
                job=job
                ))

@contextlib.contextmanager
def figure(path):
    """Context manager for saving a figure.

    Clears the current figure, yeilds, then saves the figure to the
    given path and clears the figure again.

    """
    try:
        logging.debug("Creating figure " + path)
        plt.clf()
        yield
        plt.savefig(path)
    finally:
        plt.clf()

@contextlib.contextmanager
def chdir(path):
    """Context manager for changing working directory.

    cds to the given path, yeilds, then changes back.

    >>> with chdir("/tmp"):
    >>>   # Do something in the tmp dir
    >>>   pass
    
    """
    cwd = os.getcwd()
    try:
        logging.debug("Changing cwd from " + cwd + " to " + path)
        os.chdir(path)
        yield
    finally:
        logging.debug("Changing cwd from " + path + " back to " + cwd)
        os.chdir(cwd)


def bins_uniform(num_bins, stats):
    """Returns a set of evenly sized bins for the given stats.

    Stats should be an array of statistic values, and num_bins should
    be an integer. Returns an array of bin edges, of size num_bins + 1.

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

    TODO: We should be able to make the dimensionality of this
    function flexible.

    """
    base_shape = np.shape(stats)[:-1]
    bins = np.zeros(base_shape + (num_bins + 1,))
    bins[ : -1] = sorted(stats)
    bins[-1] = np.inf
    return bins


def plot_raw_stat_hist(stats):
    """Plot the distribution of the given statistic values."""
    with figure("raw_stats"):
        plt.hist(stats, log=True, bins=250)


def setup_css(env):
    """Copy the css from 996grid/code/css into its output location."""

    src = os.path.join(os.path.dirname(REAL_PATH),
                       '996grid/code/css')

    shutil.rmtree('css', True)
    shutil.copytree(src, 'css')

    with open('css/custom.css', 'w') as out:
        template = env.get_template('custom.css')
        out.write(template.render())


def plot_conf_by_stat(conf, bins, filename='conf_by_stat'):
    with figure(filename):
        plt.plot(conf, bins[1:])
        plt.semilogy()

def plot_counts_by_stat(raw_stats, resampled_stats, bins, filename='counts_by_stat'):
    with figure(filename):
        plt.plot(bins[1:], raw_stats, label='raw')
        plt.plot(bins[1:], resampled_stats, label='resampled')
        plt.loglog()
        plt.title('Count of features by statistic')
        plt.xlabel('Statistic value')
        plt.ylabel('Features with statistic >= value')
        plt.legend()
    

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
    job.full_model = Model(schema, args.full_model)
    job.reduced_model = Model(schema, args.reduced_model)

    layout_map_full = job.full_model.layout_map()
    logging.debug("Full layout map is " + str(layout_map_full))

    logging.info("Computing coefficients")
    coeffs = find_coefficients(job.full_model, job.table)
    var_shape = job.full_model.factor_value_shape()

    layout = []
    group_keys = []
    names      = []
    (num_samples, num_features) = np.shape(job.table)
    num_groups = np.prod(var_shape)
    coeff_list = np.zeros((int(num_groups), num_features))

    for i, idx in enumerate(np.ndindex(var_shape)):
        key = job.full_model.index_num_to_sym(idx)
        group_keys.append(key)
        layout.append(layout_map_full[key])

        parts = []
        for j in range(0, len(key), 2):
            parts.append("{0}".format(key[j+1]))
        names.append("; ".join(parts))
        coeff_list[i] = coeffs[idx]

    logging.info("Computing means")
    reshaped = apply_layout(layout, job.table)
    logging.debug("Shape of reshaped is " + str(np.shape(reshaped)))
    job.means = np.mean(reshaped, axis=0)
    logging.debug("Shape of means is " + str(np.shape(job.means)))

    logging.info("Computing statistics")
    job.stats = job.stat(job.table)

    job.condition_names = names
    job.coeffs = coeff_list
    logging.debug("Condition names are" + str(job.condition_names))

    make_report(job)

    prediction = np.zeros_like(job.table)

    layout_map_red = job.reduced_model.layout_map()

    for grp in layout_map_red.values():
        data = job.table[grp]
        means = np.mean(data, axis=0)
        prediction[grp] = means

    boot = Boot(job.table, prediction, job.stat, 10000)
    ci = boot.ci()

    raw_stats = boot.raw_stats
    perm_stats = boot.permuted_stats

    plot_raw_stat_hist(raw_stats)

    bins = bins_custom(1000, raw_stats)
    raw_counts = feature_count_by_stat(raw_stats, bins)
    perm_counts   = feature_count_by_mean_stat(perm_stats, bins)
    scores = confidence_scores(raw_counts, perm_counts)

    plot_counts_by_stat(raw_counts, perm_counts, bins)
    plot_conf_by_stat(scores, bins)

    for i in range(len(raw_counts)):
        print (bins[i], raw_counts[i], perm_counts[i], scores[i])


def cumulative_hist(values, bins):
    (hist, ignore) = np.histogram(values, bins)
    return np.array(np.cumsum(hist[::-1])[::-1], float)


def feature_count_by_stat(stats, edges):
    """Count the number of features by statistic value.

    stats is a (class x feature) array where unperm_stats[class,
    feature] gives the statistic for the given feature, class
    combination. edges is a (class, bins + 1) monotonically increasing
    array where each pair of consecutive items defines the edges of a
    bin. Returns a (classes x bins) array where each item gives the
    number of features whose statistic value falls in the given bin
    for the given class.

    TODO: We should be able to make the dimensionality of this
    function flexible.
    """
    logging.debug("Shape of stats is " + str(np.shape(stats)))
    logging.debug("Shape of edges is " + str(np.shape(edges)))

    return cumulative_hist(stats, edges)

def feature_count_by_mean_stat(stats, edges):
    logging.info("Accumulating counts for permuted data")
    logging.debug("Shape of stats is " + str(np.shape(stats)))
    counts = np.zeros(np.shape(stats)[1:])

    for i, row in enumerate(stats):
        counts += cumulative_hist(row, edges)
    counts /= len(stats)
    return counts

def adjust_num_diff(V0, R):
    V = np.zeros((6,) + np.shape(V0))
    V[0] = V0
    num_ids = len(V0)
    for i in range(1, 6):
        V[i] = V[0] - V[0] / num_ids * (R - V[i - 1])
    return V[5]


def confidence_scores(raw_counts, perm_counts):
    adjusted = adjust_num_diff(perm_counts, raw_counts)
    return (raw_counts - adjusted) / raw_counts


def find_coefficients(model, data):
    factors = model.expr.variables

    layout_map = model.layout_map()    
    values = [model.schema.sample_groups([f]).keys() for f in factors]
    shape = tuple([len(v) for v in values])
    (num_samples, num_features) = np.shape(data)
    shape += (num_features,)
    coeffs = np.zeros(shape)

    sym_idx    = lambda(idx): model.index_num_to_sym(idx)
    col_nums   = lambda(idx): layout_map[sym_idx(idx)]
    group_mean = lambda(idx): np.mean(data[col_nums(idx)], axis=0)

    # Find the bias term
    bias_idx = tuple([0 for dim in shape[:-1]])
    coeffs[bias_idx] = group_mean(bias_idx)

    # Estimate the main effects
    for i, f in enumerate(factors):
        values = list(model.schema.factor_values(f))
        for j in range(1, len(values)):
            idx = np.zeros(len(shape) - 1, int)
            idx[i]= j
            idx = tuple(idx)
            coeffs[idx] = group_mean(idx) - coeffs[bias_idx]

    # Estimate the interactions between each pair of factors
    for i1 in range(len(factors)):
        f1 = factors[i1]
        vals1 = list(model.schema.factor_values(f1))

        for i2 in range(i1 + 1, len(factors)):
            f2 = factors[i2]
            vals2 = list(model.schema.factor_values(f2))
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



##############################################################################
###
### Classes
###


class ModelExpression:
    
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
            operator = ModelExpression.OP_TO_NAME[tok]
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

        return ModelExpression(prob=operator, variables=variables)

    def __str__(self):
        if len(self.variables) == 1:
            return self.variables[0]
        elif len(self.variables) > 1:
            op = ' ' + self.NAME_TO_OP[self.prob] + ' '
            return op.join(self.variables)
        else:
            return ''

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


    def layout_map(self):
        return self.schema.sample_groups(self.expr.variables)

    def index_num_to_sym(self, idx):
        schema = self.schema
        expr  = self.expr
        res = tuple()
        for i, factor in enumerate(expr.variables):
            values = list(schema.factor_values(factor))
            res += (factor, values[idx[i]])
        return res

    def factor_value_shape(self):
        schema = self.schema
        factors = self.expr.variables
        return tuple([len(schema.factor_values(f)) for f in factors])


class Job:

    DEFAULT_DIRECTORY = "pageseq_out"

    def __init__(self, 
                 directory, 
                 stat=None,
                 full_model=None,
                 reduced_model=None):
        self.directory = directory
        self.stat_name = stat
        self.full_model = full_model
        self.reduced_model = reduced_model
        self._table = None
        self._feature_ids = None

    @property
    def schema_path(self):
        """The path to the schema."""
        return os.path.join(self.directory, 'schema.yaml')

    @property
    def input_path(self):
        """Path to the input file."""
        return os.path.join(self.directory, 'input.txt')
    
    @property
    def schema(self):
        return Schema.load(open(self.schema_path), self.input_path)

    @property
    def stat(self):
        """The statistic used for this job."""
        if self.stat_name == 'f':
            return Ftest(
                layout_full=self.full_model.layout_map().values(),
                layout_reduced=self.reduced_model.layout_map().values())
        elif self.stat_name == 't':
            return Ttest(alpha=1.0)

    @property
    def table(self):
        """The data table as a (sample x feature) ndarray."""
        if self._table is None:
            self._load_table()
        return self._table

    @property
    def feature_ids(self):
        """Array of the ids of features from my input file."""
        if self._feature_ids is None:
            self._load_table()
        return self._feature_ids

    @property
    def num_features(self):
        """Number of features for this job."""
        return len(self.feature_ids)
            

    def _load_table(self):
        logging.info("Loading table from " + self.input_path)
        with open(self.input_path) as fh:

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


class Schema(object):

    def __init__(self, 
                 factors=[],
                 is_feature_id=None,
                 is_sample=None,
                 column_names=None):
        """Construct a Schema. 

  factors    - a list of allowable factors for the sample.

  column_names  - a list of strings, giving names for the columns.

  is_feature_id - a list of booleans of the same length as
                  column_names. is_feature_id[i] indicates if the ith
                  column contains feature ids (e.g. gene names).

  is_sample     - a list of booleans of the same length as
                  column_names. is_sample[i] indicates if the ith
                  column contains a sample.

Any columns for which is_feature_id is true will be treated as feature
ids, and any for which is_sample is true will be assumed to contain
intensity values. No column should have both is_feature_id and
is_sample set to true. Any columns where both is_feature_id and
is_sample are false will simply be ignored.

  """

        if column_names is None:
            raise Exception("I need column names")
        else:
            column_names = np.array(column_names)

        self.factors = collections.OrderedDict()
        self.is_feature_id = np.array(is_feature_id, dtype=bool)
        self.is_sample     = np.array(is_sample,     dtype=bool)
        self.column_names  = column_names
        self.table = None
        self.sample_to_column  = []
        self.sample_name_index = {}

        for i in range(len(self.is_sample)):
            if self.is_sample[i]:
                n = len(self.sample_to_column)
                self.sample_name_index[column_names[i]] = n
                self.sample_to_column.append(i)

        for (name, dtype) in factors:
            self.add_factor(name, dtype)

    @property
    def factor_names(self):
        """Return a list of the factor names for this schema."""

        return [x for x in self.factors]

    def sample_column_names(self):
        """Return a list of the names of columns that contain
        intensities."""

        return self.column_names[self.is_sample]

    def add_factor(self, name, dtype):
        """Add an factor with the given name and data type, which
        must be a valid numpy dtype."""
        
        default = None
        if dtype == "int":
            default = 0
        else:
            default = None

        new_table = []

        self.factors[name] = dtype

        if self.table is None:
            new_table = [(default,) for s in self.sample_column_names()]

        else:
            for row in self.table:
                row = tuple(row) + (default,)
                new_table.append(row)
            
        self.table = np.array(new_table, dtype=[(k, v) for k, v in self.factors.iteritems()])
        
    def drop_factor(self, name):
        """Remove the factor with the given name."""

        self.table = drop_fields(name)
        del self.factors[name]

    @classmethod
    def load(cls, stream, infile):
        """Load a schema from the specified stream, which must
        represent a YAML document in the format produced by
        Schema.dump. The type of stream can be any type accepted by
        yaml.load."""
        col_names = None

        if isinstance(infile, str):
            infile = open(infile)

        header_line = infile.next().rstrip()
        col_names = header_line.split("\t")

        doc = yaml.load(stream)

        # Build the arrays of column names, feature id booleans, and
        # sample booleans
        feature_id_cols = set(doc['feature_id_columns'])

        is_feature_id = [c in feature_id_cols for c in col_names]
        is_sample     = [c in doc['sample_factor_mapping'] for c in col_names]

        schema = Schema(
            column_names=col_names,
            is_feature_id=is_feature_id,
            is_sample=is_sample)

        # Now add all the factors and their types
        factors = doc['factors']
        for factor in factors:
            dtype = factor['dtype'] if 'dtype' in factor else object
            schema.add_factor(factor['name'], 
                                 dtype)

        for sample, attrs in doc['sample_factor_mapping'].iteritems():
            for name, value in attrs.iteritems():
                schema.set_factor(sample, name, value)

        return schema
    
    def save(self, out):
        """Save the schema to the specified file."""

        # Need to convert column names to strings, from whatever numpy
        # type they're stored as.
        names = [str(name) for name in self.column_names]

        sample_cols     = {}
        feature_id_cols = []

        for i, name in enumerate(names):

            if self.is_feature_id[i]:
                feature_id_cols.append(name)

            elif self.is_sample[i]:
                
                sample_cols[name] = {}
                for factor in self.factors:
                    value = self.get_factor(name, factor)
#                    if self.factors[factor].startswith("S"):
#                        value = str(value)

                    if type(value) == str:
                        pass
                    elif type(value) == np.int32:
                        value = int(value)
                    elif type(value) == np.bool_:
                        value = bool(value)
                    
                    sample_cols[name][factor] = value

        factors = []
        for name, type_ in self.factors.iteritems():
            a = { "name" : name }
            if type_ != object:
                a['dtype'] = type_
            factors.append(a)

        doc = {
            "factors"               : factors,
            "feature_id_columns"       : feature_id_cols,
            "sample_factor_mapping" : sample_cols,
            }

        data = yaml.dump(doc, default_flow_style=False, encoding=None)

        for line in data.splitlines():
            if (line == "factors:"):
                write_yaml_block_comment(out, """This lists all the factors defined for this file.
""")

            elif (line == "feature_id_columns:"):
                out.write(unicode("\n"))
                write_yaml_block_comment(out, """This lists all of the columns that contain feature IDs (for example gene ids).""")

            elif (line == "sample_factor_mapping:"):
                out.write(unicode("\n"))
                write_yaml_block_comment(out, """This sets all of the factors for all columns that represent samples. You should fill out this section to set each factor for each sample. For example, if you had a sample called sample1 that recieved some kind of treatment, and one called sample2 that did not, you might have:

sample_factor_mapping:
  sample1:
    treated: yes
  sample2:
    treated: no

""")

            out.write(unicode(line) + "\n")


    def set_factor(self, sample_name, factor, value):
        """Set an factor for a sample, identified by sample
        name."""

        sample_num = self.sample_num(sample_name)
        self.table[sample_num][factor] = value

    def get_factor(self, sample_name, factor):
        """Get an factor for a sample, identified by sample
        name."""

        sample_num = self.sample_num(sample_name)
        value = self.table[sample_num][factor]
#        if self.factors[factor].startswith("S"):
#            value = str(value)
        return value

    def sample_num(self, sample_name):
        """Return the sample number for sample with the given
        name. The sample number is the index into the table for the
        sample."""

        return self.sample_name_index[sample_name]

    def sample_groups(self, factors):
        """Returns a dictionary mapping each value of factor to the
        list of sample numbers that have that value set."""

        grouping = {}

        for i, row in enumerate(self.table):
            key = []
            for f in factors:
                key.append(f)
                key.append(self.table[f][i])
            key = tuple(key)
            if key not in grouping:
                grouping[key] = []
            grouping[key].append(i)

        return grouping

    def condition_name(self, c):
        """Return a name for condition c, based on the factor values for that condition"""
        pass

    def factor_values(self, factor):
        values = set()
        for sample in self.sample_column_names():
            values.add(self.get_factor(sample, factor))
        return values


def write_yaml_block_comment(fh, comment):
    result = ""
    for line in comment.splitlines():
        result += textwrap.fill(line, initial_indent = "# ", subsequent_indent="# ")
        result += "\n"
    fh.write(unicode(result))


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
        logging.info("Computing statistics based on sampled residuals")

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
        logging.info("Computing confidence intervals")
        intervals = scipy.stats.mstats.mquantiles(
            self.permuted_stats, prob=levels, axis=0)
        return intervals


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
        out = open(job.schema_path, mode)
        schema.save(out)
    except IOError as e:
        if e.errno == errno.EEXIST:
            raise UsageException("The schema file \"{}\" already exists. If you want to overwrite it, use the --force or -f argument.".format(filename))
        raise e

    logging.info("Copying {0} to {1}".format(
        infile.name,
        job.input_path))
    shutil.copyfile(infile.name, job.input_path)
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
            filename=job.schema_path)

class Step:
    def __init__(self, name, forward, reverse):
        

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


class Ttest:

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
    
    def __init__(self, alpha):
        self.alpha = alpha

        if isinstance(alpha, numbers.Number):
            self.children = None
        else:
            self.children = [Ttest(a) for a in alpha]

    @classmethod
    def compute_s(cls, data):
        """
        v1 and v2 should have the same number of rows.
        """
                
        var = np.var(data, ddof=1, axis=1)
        size = ma.count(data, axis=1)
        return np.sqrt(np.sum(var * size, axis=0) / np.sum(size, axis=0))


    @classmethod
    def find_default_alpha(cls, table):
        """
        Return a default value for alpha. 
        
        Table should be an ndarray, with shape (conditions, samples, features).
        
        """

        alphas = np.zeros(len(table))
        (num_classes, samples_per_class, num_features) = np.shape(table)

        for c in range(1, num_classes):
            subset = table[([c, 0],)]
            values = cls.compute_s(subset)
            mean = np.mean(values)
            residuals = values[values < mean] - mean
            sd = np.sqrt(sum(residuals ** 2) / (len(residuals) - 1))
            alphas[c] = mean * 2 / np.sqrt(samples_per_class * 2)

        return alphas


    def __call__(self, data):
        """Computes the t-stat.

        Input must be an ndarray with at least 2 dimensions. Axis 0
        should be class, and axis 1 should be sample. If there are
        more than two axes, the t-stat will be vectorized to all axes
        past axis .
        """

        class_axis = 0
        sample_axis = 1

        n = ma.count(data, axis=1)
        n1 = n[0]
        n2 = n[1]

        # Variance for each row of v1 and v2 with one degree of
        # freedom. var1 and var2 will be 1-d arrays, one variance for each
        # feature in the input.
        var   = np.var(data, ddof=1, axis=sample_axis)
        means = np.mean(data, axis=sample_axis)
        prod  = var * (n - 1)
        S     = np.sqrt((prod[0] + prod[1]) / (n1 + n2 - 2))
        numer = (means[0] - means[1]) * np.sqrt(n1 * n2)
        denom = (self.alpha + S) * np.sqrt(n1 + n2)

        return numer / denom


class Ftest:

    def __init__(self, layout_full, layout_reduced):
        self.layout_full = layout_full
        self.layout_reduced = layout_reduced


    def __call__(self, data):
        """Compute the f-test for the given ndarray.

        Input must have 2 or more dimensions. Axis 0 must be sample,
        axis 1 must be condition. Operations are vectorized over any
        subsequent axes. So, for example, an input array with shape
        (3, 2) would represent 1 feature for 2 conditions, each with
        at most 3 samples. An input array with shape (5, 3, 2) would
        be 5 features for 3 samples of 2 conditions.

        TODO: Make sure masked input arrays work.

        """

        data_full = apply_layout(self.layout_full, data)
        data_red  = apply_layout(self.layout_reduced,  data)
    
        # Means for the full and reduced model
        y_full = np.mean(data_full, axis=0)
        y_red  = np.mean(data_red,  axis=0)

        # Degrees of freedom
        p_red  = len(self.layout_reduced)
        p_full = len(self.layout_full)
        n = len(self.layout_reduced) * len(self.layout_reduced[0])

        # Residual sum of squares for the reduced and full model
        rss_red  = double_sum((data_red  - y_red)  ** 2)
        rss_full = double_sum((data_full - y_full) ** 2)

        numer = (rss_red - rss_full) / (p_full - p_red)
        denom = rss_full / (n - p_full)
        return numer / denom


if __name__ == '__main__':
    main()
