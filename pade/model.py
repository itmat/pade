"""Classes used to model PADE data and jobs."""

from __future__ import absolute_import, print_function, division

import numpy as np
from collections import namedtuple, OrderedDict
from itertools import product

from StringIO import StringIO
import tokenize
import logging
import csv
import yaml
import textwrap

DEFAULT_STATISTIC = 'f_test'
DEFAULT_NUM_BINS = 1000
DEFAULT_NUM_SAMPLES = 1000
DEFAULT_SAMPLE_WITH_REPLACEMENT = False
DEFAULT_SAMPLE_FROM_RESIDUALS = False
DEFAULT_MIN_CONF = 0.25
DEFAULT_CONF_INTERVAL = 0.05
DEFAULT_EQUALIZE_MEANS = True
DEFAULT_TUNING_PARAMS=[0.001, 0.01, 0.1, 1, 3, 10, 30, 100, 300, 1000, 3000]

TableWithHeader = namedtuple('TableWithHeader', ['header', 'table'])

class ModelExpressionException(Exception):
    """Thrown when a model expression is invalid."""
    pass


class ModelExpression:
    """Represents a list of variables and an operator."""


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
        for factor in self.expr.variables:
            if factor not in self.schema.factors:
                raise Exception(
                    "Factor '" + factor + 
                    "' is not defined in the schema. Valid factors are " + 
                    str(self.schema.factors))

    @property
    def layout(self):
        s = self.schema
        return [s.indexes_with_assignments(a)
                for a in s.possible_assignments(self.expr.variables)]


class Summary(object):
    """Summarizes the results of a job by confidence level."""

    def __init__(self, bins, best_param_idxs, counts):
        self.bins = bins
        self.best_param_idxs = best_param_idxs
        self.counts = counts

class Input(object):

    """Raw(ish) input for the job.

    Consists only of a 2d array of the intensity values, and a 1d
    array of feature ids. Grouping if the columns is handled elsewhere.

    """
    def __init__(self, table, feature_ids):
        assert_ndarray(table, name='table', ndim=2)
        assert_ndarray(feature_ids, name='feature_ids', ndim=1)
        
        self.table = table
        """The raw data

        (feature x sample) ndarray.

        """

        self.feature_ids = feature_ids
        """Array of feature ids"""

    @classmethod
    def from_raw_file(cls, path, schema, limit=None):
        """Load the given input file into memory.

        :param path:
          Path to an input file, which must be a tab-delimited file
          with a header line.
        
        """

        ids = []
        table = []

        log_interval = 100000

        with open(path) as fh:
            csvfile = csv.DictReader(fh, delimiter="\t")
            headers = csvfile.fieldnames

            sample_names = schema.sample_column_names

            for i, rec in enumerate(csvfile):
                if limit is not None and i > limit:
                    break
                table.append([ float(rec[name]) for name in sample_names ])
                ids.append(rec[schema.feature_id_column_names[0]])

                if (i % log_interval) == log_interval - 1:
                    logging.debug("Copied {0} rows".format(i + 1))

        table = np.array(table)
        ids   = np.array(ids)

        logging.debug(
            "Input has {features} features and {samples} samples".format(
                features=np.size(table, 0),
                samples=np.size(table, 1)))

        return Input(table, ids)



class Settings:

    """The settings that control how the job is run."""

    def __init__(
        self,
        stat_name=DEFAULT_STATISTIC,
        num_bins=DEFAULT_NUM_BINS,
        num_samples=DEFAULT_NUM_SAMPLES,
        sample_from_residuals=DEFAULT_SAMPLE_FROM_RESIDUALS,
        sample_with_replacement=DEFAULT_SAMPLE_WITH_REPLACEMENT,
        condition_variables=None,
        block_variables=None,
        min_conf=DEFAULT_MIN_CONF,
        equalize_means=DEFAULT_EQUALIZE_MEANS,
        conf_interval=DEFAULT_CONF_INTERVAL,
        tuning_params=DEFAULT_TUNING_PARAMS,
        equalize_means_ids=None):

        self.stat_name = stat_name
        """Name of the statistic to use."""

        self.num_samples = num_samples
        """Max number of samples to use for permutation test or bootstrapping"""

        self.sample_from_residuals = sample_from_residuals
        """If true, sample from residuals rather than raw values."""

        self.sample_with_replacement = sample_with_replacement
        """If true, do sampling with replacement (bootstrapping).

        If false, do permutation test.

        """

        self.num_bins = num_bins
        """Number of bins used to discretize statistic space"""

        self.block_variables = block_variables
        """List of "blocking" or "nuisance" variables."""

        self.condition_variables = condition_variables
        """List of variables you want to test for differential effects."""

        self.tuning_params = tuning_params
        """Optional list of tuning parameters for statistic."""

        self.equalize_means = equalize_means
        """If true, shift the values so the mean of each group is 0."""

        self.min_conf = min_conf
        """Minimum confidence level to report on."""

        self.conf_interval = conf_interval
        """Interval of confidence values to report on."""

        self.equalize_means_ids = equalize_means_ids
        """List of ids of features to equalize means for."""


class Results:
    """The bulk of the results of the job."""
    
    def __init__(self):
    
        self.bins = None
        self.bin_to_unperm_count = None
        self.bin_to_mean_perm_count = None
        self.bin_to_score = None
        self.feature_to_score = None
        self.raw_stats = None
        self.sample_indexes = None
        self.group_means = None
        self.coeff_values = None
        self.fold_change = None
        self.order_by_foldchange_original = None
        self.order_by_score_original = None


class Job:

    """Bundle of things that make up a PADE analysis.
    
    Contains the raw input, the schema describing the input, the
    settings that control how the job was executed, the results of the
    job, and a summary of those results.

    """
    def __init__(self, 
                 job_id=None,
                 input=None,
                 schema=None,
                 settings=None,
                 results=None,
                 summary=None):
        
        self.job_id   = job_id
        self.input    = input
        self.settings = settings
        self.schema   = schema
        self.results  = results
        self.summary  = summary

    def layout(self, variables):
        s = self.schema
        assignments = s.possible_assignments(variables)
        result = [s.indexes_with_assignments(a) for a in assignments]
        seen = set()
        for grp in result:
            for idx in grp:
                if idx in seen:
                    raise Exception("The schema seems be be corrupt, because " +
                                    "column " + str(idx) + " appears more " +
                                    "than once in the layout defined by " +
                                    "variables " + str(variables) + 
                                    " with assignments " + str(assignments))
                seen.add(idx)
        return result

    @property
    def full_model(self):
        return Model(self.schema, "*".join(self.settings.block_variables + self.settings.condition_variables))

    @property
    def reduced_model(self):
        return Model(self.schema, "*".join(self.settings.block_variables))

    @property
    def condition_layout(self):
        return self.layout(self.settings.condition_variables)

    @property
    def block_layout(self):
        return self.layout(self.settings.block_variables)

    @property
    def full_layout(self):
        return self.layout(self.settings.block_variables + self.settings.condition_variables)

    @property
    def full_variables(self):
        return self.settings.block_variables + self.settings.condition_variables


def assert_ndarray(array, name=None, ndim=None):
    if ndim is not None:
        if array.ndim != ndim:
            msg = ("Array argument {name} must be {ndim}-dimensional, " +
                   "but it has shape {shape}")
            raise Exception(msg.format(
                    name=name,
                    ndim=ndim,
                    shape=array.shape))
                
def write_yaml_block_comment(fh, comment):
    result = ""
    for line in comment.splitlines():
        result += textwrap.fill(line, initial_indent = "# ", subsequent_indent="# ")
        result += "\n"
    fh.write(unicode(result))


class Schema(object):

    """Describes a PADE input file.

    """

    def __init__(self, 
                 column_names=None,
                 column_roles=None):

        """Construct a Schema. 

        :param column_names:
          List of strings, giving names for the columns.

        :param is_feature_id:
          List of booleans of the same length as
          column_names. is_feature_id[i] indicates if the ith column
          contains feature ids (e.g. gene names).

        :param is_sample:
          List of booleans of the same length as
          column_names. is_sample[i] indicates if the ith column
          contains a sample.

          Any columns for which is_feature_id is true will be treated
          as feature ids, and any for which is_sample is true will be
          assumed to contain intensity values. No column should have
          both is_feature_id and is_sample set to true. Any columns
          where both is_feature_id and is_sample are false will simply
          be ignored.

          """

        self.factor_values = OrderedDict()

        self.sample_to_factor_values = OrderedDict()
        """Maps a column name to a dict which maps factor name to value."""

        self.column_roles = None
        """Nth item is true if Nth column is a sample."""

        self.column_names  = None
        """List of column names."""

        self.sample_name_index = {}

        if column_names is not None:
            self.set_columns(column_names, column_roles)

    @property
    def factors(self):
        """List of the factor names for this schema."""
        return self.factor_values.keys()
            
    def set_columns(self, names, roles):
        self.sample_to_factor_values.clear()
        self.column_roles = np.array(roles)
        self.column_names = np.array(names)
        self.sample_name_index = {}
        logging.info("Setting sample name index, which is " + 
                     str(self.sample_name_index) + " of length " + 
                     str(len(self.sample_name_index)))
        for i, name in enumerate(names):
            if roles[i] == 'sample':
                self.sample_name_index[name] = len(self.sample_name_index)
                self.sample_to_factor_values[name] = { f : None for f in self.factors }
            

    @property
    def sample_column_names(self):
        """List of the names of columns that contain intensities."""
        return self.column_names[self.column_roles == 'sample']

    @property
    def feature_id_column_names(self):
        """Name of column that contains feature ids."""
        return self.column_names[self.column_roles == 'feature_id']

    def _check_factors(self, factors):

        if factors is None:
            factors = self.factors

        factors = list(factors)
        correct_order = [f for f in self.factors if f in factors]
        
#        if factors != correct_order:
#            raise Exception("Please request factors in correct order: "
#                            + str(correct_order) + ", not " + str(factors))

        return factors

    def factor_combinations(self, factors=None):
        """Return list of possible combinations of values for given factors.

        Each item of the returned list is a tuple of values, where
        value i is one of the values defined in the schema for factor i.
        """

        factors = self._check_factors(factors)
        values = [self.factor_values[f] for f in factors]
        return list(product(*values))

    def are_baseline(self, assignments):
        return [ v == self.baseline_value(f) for (f, v) in assignments.items() ]

    def baseline_value(self, factor):
        """Return the first value listed in the schema for given factor."""
        return self.factor_values[factor][0]

    def baseline_values(self, factors):
        """Return first value defined for each of the factors given."""        
        return [self.baseline_value(f) for f in factors]

    def has_baseline(self, assignment):
        """Return true if any of the variables is assigned to their baseline.

        assignment must be a dictionary mapping factors in this schema to
        a valid value for that factor. Returns true if any of those mappings
        is to the 'baseline' value for that factor.

        """
        return any(self.are_baseline(assignment))

    def sample_matches_assignments(self, sample_name, assignments):
        """Return true if the sample with the given has the given assignments.

        sample_name - must be the name of one of the samples in this schema.
        assignments - must be a mapping from factor name to value

        """
        matches = lambda f: self.get_factor(sample_name, f) == assignments[f]
        return all(map(matches, assignments))

    def samples_with_assignments(self, assignments):
        """Return list of sample names that have the given assignments.

        assignments - must be a mapping from factor name to value

        """
        names = self.sample_column_names
        matches = lambda x: self.sample_matches_assignments(x, assignments)
        return filter(matches, names)

    def indexes_with_assignments(self, assignments):
        """Return list of indexes that have the given assignments.
        
        assignments - must be a mapping from factor name to value

        """
        samples = self.samples_with_assignments(assignments)
        indexes = [self.sample_num(s) for s in samples]
        return indexes

    def possible_assignments(self, factors=None):
        """Return a list of all possible mappings from factor name to value.

        Optionally provide the factors argument to limit combinations
        to those factors.

        """
        factors = self._check_factors(factors)
        return [
            OrderedDict(zip(factors, values))
            for values in self.factor_combinations(factors)]

    def add_factor(self, name, values=[]):
        """Add a factor with the given name and values."""
        self.factor_values[name] = list(values)
        for sample in self.sample_to_factor_values:
            self.sample_to_factor_values[sample][name] = None

    def remove_factor(self, factor):
        del self.factor_values[factor]
        for sample in self.sample_to_factor_values:
            del self.sample_to_factor_values[factor]

    @classmethod
    def load(cls, stream):
        """Load a schema from the specified stream.

        The stream can be any type accepted by yaml.load, and must
        represent a YAML document in the format produced by
        Schema.dump.

        """
        logging.info("Loading schema from " + str(stream))
        doc = yaml.load(stream)

        if doc is None:
            raise Exception("Didn't find a YAML schema document in " + str(stream))

        col_names = doc['headers']

        # Build the arrays of column names, feature id booleans, and
        # sample booleans
        feature_id_cols = set(doc['feature_id_columns'])

        roles = []
        for c in col_names:
            if c in feature_id_cols:
                role = 'feature_id'
            elif c in doc['sample_factor_mapping']:
                role = 'sample'
            else:
                role = None
            roles.append(role)

        schema = Schema(
            column_names=col_names,
            column_roles=roles)

        # Now add all the factors and their types
        factors = doc['factors']
        for factor in factors:
            schema.add_factor(factor['name'], factor['values'])

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

            if self.column_roles[i] == 'feature_id':
                feature_id_cols.append(name)

            elif self.column_roles[i] == 'sample':

                sample_cols[name] = {}
                for factor in self.factors:
                    if factor in self.sample_to_factor_values[name]:
                        value = self.get_factor(name, factor)
                        sample_cols[name][factor] = value

        factors = []
        for name, values in self.factor_values.items():
            factors.append({ "name" : name, "values" : values })

        doc = {
            'headers'               : names,
            "factors"               : factors,
            "feature_id_columns"    : feature_id_cols,
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

            elif (line == "headers:"):
                out.write(unicode("\n"))
                write_yaml_block_comment(out, """These are the headers in the input file. YOU MUST NOT CHANGE THESE!""")

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
        """Set a factor for a sample, identified by sample name."""
        allowed = self.factor_values[factor]
        if value not in allowed:
            raise Exception("""\
                Value '{value}' is not allowed for factor {factor};
                allowable values are {allowed}.
                """.format(value=value, factor=factor, allowed=allowed))

        self.sample_to_factor_values[sample_name][factor] = value

    def get_factor(self, sample_name, factor):
        """Get an factor for a sample, identified by sample
        name."""
        if sample_name not in self.sample_to_factor_values:
            raise Exception("No sample called " + str(sample_name) + ". " +
                            "The samples I have are " + 
                            str(self.sample_to_factor_values.keys()))
                            

        return self.sample_to_factor_values[sample_name][factor]


    def sample_num(self, sample_name):
        """Return the sample number for sample with the given
        name. The sample number is the index into the table for the
        sample."""

        return self.sample_name_index[sample_name]
