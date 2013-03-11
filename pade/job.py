"""Handles job state persistence."""

import logging
import h5py
import collections
import csv

from StringIO import StringIO
from pade.schema import Schema
from pade.model import Model
from pade.common import *
import numpy as np

DEFAULT_STATISTIC = 'f_test'
DEFAULT_NUM_BINS = 1000
DEFAULT_NUM_SAMPLES = 1000
DEFAULT_SAMPLE_WITH_REPLACEMENT = False
DEFAULT_SAMPLE_FROM_RESIDUALS = False
DEFAULT_MIN_CONF = 0.25
DEFAULT_CONF_INTERVAL = 0.05
DEFAULT_EQUALIZE_MEANS = True
DEFAULT_TUNING_PARAMS=[0.001, 0.01, 0.1, 1, 3, 10, 30, 100, 300, 1000, 3000]

TableWithHeader = collections.namedtuple('TableWithHeader', ['header', 'table'])

class Summary(object):
    def __init__(self, bins, best_param_idxs, counts):
        self.bins = bins
        self.best_param_idxs = best_param_idxs
        self.counts = counts

class Input(object):

    """Raw(ish) input for the job."""
    
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


def load_input(db):
    return Input(db['table'][...],
                 db['feature_ids'][...])

def load_job(path):

    with h5py.File(path, 'r') as db:
        return Job(
            settings=load_settings(db),
            input=load_input(db),
            schema=load_schema(db),
            results=load_results(db),
            summary=load_summary(db))


def load_settings(db):

    if 'equalize_means_ids' in db:
        equalize_means_ids = db['equalize_means_ids'][...]
    else:
        equalize_means_ids = None

    return Settings(
        stat_name = db.attrs['stat_name'],
        num_bins = db.attrs['num_bins'],
        num_samples = db.attrs['num_samples'],
        sample_from_residuals = db.attrs['sample_from_residuals'],
        sample_with_replacement = db.attrs['sample_with_replacement'],
        condition_variables = list(db.attrs['condition_variables']),
        block_variables = list(db.attrs['block_variables']),
        min_conf = db.attrs['min_conf'],
        conf_interval = db.attrs['conf_interval'],
        tuning_params = db['tuning_params'][...],
        equalize_means_ids = equalize_means_ids)


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


def save_input(input, db):
    ids = input.feature_ids
    # Saving feature ids is tricky because they are strings
    dt = h5py.special_dtype(vlen=str)
    db.create_dataset("table", data=input.table)
    db.create_dataset("feature_ids", (len(ids),), dt)
    for i, fid in enumerate(ids):
        input.feature_ids[i] = fid

def save_schema(schema, db):
    schema_str = StringIO()
    schema.save(schema_str)
    db.attrs['schema'] = str(schema_str.getvalue())


def save_settings(settings, db):
        
    db.create_dataset("tuning_params", data=settings.tuning_params)
    db.attrs['stat_name'] = settings.stat_name
    db.attrs['num_bins'] = settings.num_bins
    db.attrs['num_samples'] = settings.num_samples
    db.attrs['sample_from_residuals'] = settings.sample_from_residuals
    db.attrs['sample_with_replacement'] = settings.sample_with_replacement
    db.attrs['condition_variables'] = settings.condition_variables
    db.attrs['block_variables'] = settings.block_variables
    db.attrs['min_conf'] = settings.min_conf
    db.attrs['conf_interval'] = settings.conf_interval

    if settings.equalize_means_ids is not None:
        db['equalize_means_ids'] = settings.equalize_means_ids


def save_table(db, table, name):
    db.create_dataset(name, data=table.table)
    db[name].attrs['headers'] = table.header        

    
def save_job(path, job):
    with h5py.File(path, 'w') as db:
        save_input(job.input, db)
        save_schema(job.schema, db)
        save_settings(job.settings, db)
        save_results(job.results, db)
        save_summary(job.summary, db)
    db.close()


def load_table(db, name):
    if name in db:
        ds = db[name]
        return TableWithHeader(ds.attrs['headers'], ds[...])
    else:
        return None


def load_schema(db):
    schema_str = StringIO(db.attrs['schema'])
    return Schema.load(schema_str)

def load_summary(db):
    if 'summary' in db:
        return Summary(
            db['summary']['bins'][...],
            db['summary']['best_param_idxs'][...],
            db['summary']['counts'][...])
    else:
        return None


def load_results(db):

    results = Results()

    if 'bins' in db:
        results.bins = db['bins'][...]
    if 'bin_to_unperm_count' in db:
        results.bin_to_unperm_count    = db['bin_to_unperm_count'][...]
    if 'bin_to_mean_perm_count' in db:
        results.bin_to_mean_perm_count = db['bin_to_mean_perm_count'][...]
    if 'bin_to_score' in db:
        results.bin_to_score           = db['bin_to_score'][...]
    
    if 'feature_to_score' in db:
        results.feature_to_score = db['feature_to_score'][...]
    
    if 'raw_stats' in db:
        results.raw_stats = db['raw_stats'][...]

    if 'sample_indexes' in db:
        results.sample_indexes = db['sample_indexes'][...]

    # Group means, coefficients, and fold change, with the header information
    results.group_means  = load_table(db, 'group_means')
    results.coeff_values = load_table(db, 'coeff_values')
    results.fold_change  = load_table(db, 'fold_change')
    # Orderings
    if 'orderings' in db:
        results.ordering_by_score_original      = db['orderings']['by_score_original'][...]
        results.ordering_by_foldchange_original = db['orderings']['by_foldchange_original'][...]

    return results

class Job:

    """Interface for the HDF5 file that we use to persist the job state."""

    def __init__(self, 
                 input=None,
                 schema=None,
                 settings=None,
                 results=None,
                 summary=None):
        
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


