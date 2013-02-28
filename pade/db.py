"""Handles job state persistence."""

import logging
import h5py
import collections

from StringIO import StringIO
from pade.schema import Schema
from pade.model import Model
import numpy as np

TableWithHeader = collections.namedtuple('TableWithHeader', ['header', 'table'])

class Input(object):

    """Raw(ish) input for the job."""
    
    def __init__(self):
        self.table = None
        self.feature_ids = None


    def import_table(self, path):
        """Load the given input file into memory.

        :param path:
          Path to an input file, which must be a tab-delimited file
          with a header line.
        
        """
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

        self.table = table
        self.feature_ids = ids

    


class Settings:

    """The settings that control how the job is run."""

    def __init__(self):
        # Settings
        self.stat_name = None
        self.num_bins = None
        self.num_samples = None
        self.sample_from_residuals = None
        self.sample_with_replacement = None
        self.block_variables    = None
        self.condition_variables = None
        self.tuning_params = None
        self.equalize_means = None
        self.min_conf = None
        self.conf_interval = None

def save_input(input, db):
    ids = input.feature_ids
    # Saving feature ids is tricky because they are strings
    dt = h5py.special_dtype(vlen=str)
    input.table       = db.create_dataset("table", data=input.table)
    input.feature_ids = db.create_dataset("feature_ids", (len(ids),), dt)
    for i, fid in enumerate(ids):
        input.feature_ids[i] = fid



def save_settings(settings, db):
        
    db.create_dataset("tuning_params", data=settings.tuning_params)
    db.attrs['stat_name'] = settings.stat
    db.attrs['num_bins'] = settings.num_bins
    db.attrs['num_samples'] = settings.num_samples
    db.attrs['sample_from_residuals'] = settings.sample_from_residuals
    db.attrs['sample_with_replacement'] = settings.sample_with_replacement
    db.attrs['condition_variables'] = settings.condition_variables
    db.attrs['block_variables'] = settings.block_variables
    db.attrs['min_conf'] = settings.min_conf
    db.attrs['conf_interval'] = settings.conf_interval

def load_input(self):
    input = Input()
    input.table = self.file['table'][...]
    input.feature_ids = self.file['feature_ids'][...]
    return input


def load_settings(db):
    s = Settings()
    s.stat_name = db.attrs['stat_name']
    s.num_bins = db.attrs['num_bins']
    s.num_samples = db.attrs['num_samples']
    s.sample_from_residuals = db.attrs['sample_from_residuals']
    s.sample_with_replacement = db.attrs['sample_with_replacement']
    s.condition_variables = list(db.attrs['condition_variables'])
    s.block_variables = list(db.attrs['block_variables'])
    s.min_conf = db.attrs['min_conf']
    s.conf_interval = db.attrs['conf_interval']
    s.tuning_params = db['tuning_params'][...]
    return s



class DB:

    """Interface for the HDF5 file that we use to persist the job state."""

    def __init__(self, 
                 schema=None,
                 schema_path=None,
                 path=None):

        self.settings = None

        self.input = Input()

        self.schema_path = schema_path
        self.path = path

        # Results
        self.bins = None
        self.bin_to_unperm_count = None
        self.bin_to_mean_perm_count = None
        self.bin_to_score = None
        self.feature_to_score = None
        self.raw_stats = None
        self.summary_counts = None
        self.best_param_idxs = None
        self.sample_indexes = None
        self.group_means = None
        self.coeff_values = None
        self.fold_change = None

        self.file = None
        self.schema = schema


    def save(self):
        logging.info("Saving job results to " + self.path)
        f = h5py.File(self.path, 'w')
        self.file = f

        save_input(self.input, self.file)
        save_settings(self.settings, self.file)

        schema_str = StringIO()
        self.schema.save(schema_str)

        self.bins = f.create_dataset("bins", data=self.bins)

        f.attrs['schema'] = str(schema_str.getvalue())

        self.bin_to_mean_perm_count = f.create_dataset("bin_to_mean_perm_count", data=self.bin_to_mean_perm_count)
        self.bin_to_unperm_count   = f.create_dataset("bin_to_unperm_count", data=self.bin_to_unperm_count)
        self.bin_to_score = f.create_dataset("bin_to_score", data=self.bin_to_score)
        self.feature_to_score = f.create_dataset("feature_to_score", data=self.feature_to_score)
        self.raw_stats = f.create_dataset("raw_stats", data=self.raw_stats)

        summary = f.create_group('summary')
        summary['bins']            = self.summary_bins
        summary['counts']          = self.summary_counts
        summary['best_param_idxs'] = self.best_param_idxs

        self.sample_indexes = f.create_dataset("sample_indexes", data=self.sample_indexes)

        self.save_table(self.group_means, 'group_means')
        self.save_table(self.fold_change, 'fold_change')
        self.save_table(self.coeff_values, 'coeff_values')

        self.compute_orderings()
        self.file = None

        f.close()

    def compute_orderings(self):

        grp = self.file.create_group('orderings')
        original = np.arange(len(self.input.feature_ids))
        stats = self.feature_to_score[...]
        rev_stats = 0.0 - stats

        by_score_original = np.zeros(np.shape(self.raw_stats), int)
        for i in range(len(self.settings.tuning_params)):
            by_score_original[i] = np.lexsort(
                (original, rev_stats[i]))

        grp['score_original'] = by_score_original

        by_foldchange_original = np.zeros(np.shape(self.fold_change.table), int)
        foldchange = self.fold_change.table[...]
        rev_foldchange = 0.0 - foldchange
        for i in range(len(self.fold_change.header)):
            keys = (original, rev_foldchange[..., i])

            by_foldchange_original[..., i] = np.lexsort(keys)

        grp['foldchange_original'] = by_foldchange_original

    def save_table(self, table, name):
        self.file.create_dataset(name, data=table.table)
        self.file[name].attrs['headers'] = table.header        
    
    def load_table(self, name):
        ds = self.file[name]
        if ds is None:
            raise Exception("No dataset called " + str(name)) 
        return TableWithHeader(ds.attrs['headers'], ds[...])

    def load(self):

        logging.info("Loading job results from " + self.path)
        file = None
        try:
            file = h5py.File(self.path, 'r')
        except IOError as e:
            raise IOError("While trying to load database from " + self.path, e)

        self.file = file

        self.settings = load_settings(self.file)
        self.input    = load_input(self.file)
        self.bins = file['bins'][...]
        self.bin_to_unperm_count    = file['bin_to_unperm_count'][...]
        self.bin_to_mean_perm_count = file['bin_to_mean_perm_count'][...]
        self.bin_to_score           = file['bin_to_score'][...]

        self.feature_to_score = file['feature_to_score'][...]
        self.raw_stats = file['raw_stats'][...]

        # Summary counts by bin, based on optimal tuning params at each level
        self.summary_bins    = file['summary']['bins'][...]
        self.summary_counts  = file['summary']['counts'][...]
        self.best_param_idxs = file['summary']['best_param_idxs'][...]

        self.sample_indexes = file['sample_indexes'][...]

        # Group means, coefficients, and fold change, with the header information
        self.group_means  = self.load_table('group_means')
        self.coeff_values = self.load_table('coeff_values')
        self.fold_change  = self.load_table('fold_change')

        # Orderings
        self.ordering_by_score_original      = file['orderings']['score_original'][...]
        self.ordering_by_foldchange_original = file['orderings']['foldchange_original'][...]

        schema_str = StringIO(file.attrs['schema'])
        self.schema = Schema.load(schema_str)

        file.close()
        logging.info("Done loading results")


    def layout(self, variables):
        s = self.schema
        return [s.indexes_with_assignments(a)
                for a in s.possible_assignments(variables)]

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

