import logging
import h5py
import collections

from StringIO import StringIO
from pade.schema import Schema
from pade.model import Model
import numpy as np

TableWithHeader = collections.namedtuple('TableWithHeader', ['header', 'table'])

class Settings:

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

class DB:

    def __init__(self, 
                 schema=None,
                 schema_path=None,
                 path=None):

        self.settings = None

        # This will be populated lazily by the table and feature_ids
        # properties
        self.table       = None
        self.feature_ids = None
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

    def save_settings(self):
        
        file = self.file
        s = self.settings
        file.create_dataset("tuning_params", data=s.tuning_params)
        file.attrs['stat_name'] = s.stat
        file.attrs['num_bins'] = s.num_bins
        file.attrs['num_samples'] = s.num_samples
        file.attrs['sample_from_residuals'] = s.sample_from_residuals
        file.attrs['sample_with_replacement'] = s.sample_with_replacement
        file.attrs['condition_variables'] = s.condition_variables
        file.attrs['block_variables'] = s.block_variables
        file.attrs['min_conf'] = s.min_conf
        file.attrs['conf_interval'] = s.conf_interval

    def load_settings(self):
        s = Settings()
        file = self.file
        s.stat_name = file.attrs['stat_name']
        s.num_bins = file.attrs['num_bins']
        s.num_samples = file.attrs['num_samples']
        s.sample_from_residuals = file.attrs['sample_from_residuals']
        s.sample_with_replacement = file.attrs['sample_with_replacement']
        s.condition_variables = list(file.attrs['condition_variables'])
        s.block_variables = list(file.attrs['block_variables'])
        s.min_conf = file.attrs['min_conf']
        s.conf_interval = file.attrs['conf_interval']
        s.tuning_params = file['tuning_params'][...]
        self.settings = s

    def save(self):
        logging.info("Saving job results to " + self.path)
        file = h5py.File(self.path, 'r+')
        self.file = file
        # Saving feature ids is tricky because they are strings
        dt = h5py.special_dtype(vlen=str)
        ids = self.feature_ids
        self.feature_ids = file.create_dataset("feature_ids", (len(ids),), dt)
        for i, fid in enumerate(ids):
            self.feature_ids[i] = fid

        schema_str = StringIO()
        self.schema.save(schema_str)

        self.table = file.create_dataset("table", data=self.table)
        self.bins = file.create_dataset("bins", data=self.bins)

        # Settings:
        self.save_settings()

        file.attrs['schema'] = str(schema_str.getvalue())

        self.bin_to_mean_perm_count = file.create_dataset("bin_to_mean_perm_count", data=self.bin_to_mean_perm_count)
        self.bin_to_unperm_count   = file.create_dataset("bin_to_unperm_count", data=self.bin_to_unperm_count)
        self.bin_to_score = file.create_dataset("bin_to_score", data=self.bin_to_score)
        self.feature_to_score = file.create_dataset("feature_to_score", data=self.feature_to_score)
        self.raw_stats = file.create_dataset("raw_stats", data=self.raw_stats)

        summary = file.create_group('summary')
        summary['bins']            = self.summary_bins
        summary['counts']          = self.summary_counts
        summary['best_param_idxs'] = self.best_param_idxs

        self.sample_indexes = file.create_dataset("sample_indexes", data=self.sample_indexes)

        self.save_table(self.group_means, 'group_means')
        self.save_table(self.fold_change, 'fold_change')
        self.save_table(self.coeff_values, 'coeff_values')

        self.file = file
        self.compute_orderings()
        self.file = None

        file.close()

    def compute_orderings(self):

        grp = self.file.create_group('orderings')
        original = np.arange(len(self.feature_ids))
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

        self.load_settings()

        self.table = file['table'][...]
        self.feature_ids = file['feature_ids'][...]
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


