"""Handles job state persistence."""

import logging
import h5py
import collections
import csv
import numpy as np

from StringIO import StringIO

from pade.model import Model
from pade.common import assert_ndarray


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


