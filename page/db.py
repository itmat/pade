import logging
import h5py
from StringIO import StringIO
from page.schema import Schema
from page.model import Model

class DB:

    def __init__(self, 
                 schema=None,
                 schema_path=None,
                 path=None,
                 input_txt_path=None):

        # This will be populated lazily by the table and feature_ids
        # properties
        self.table       = None
        self.feature_ids = None
        self.schema_path = schema_path
        self.path = path

        # Settings
        self.stat_name = None
        self.min_conf = None
        self.conf_levels = None
        self.num_bins = None
        self.num_samples = None
        self.sample_from = None
        self.sample_method = None
        self.full_model    = None
        self.reduced_model = None
        self.sample_indexes = None
        self.equalize_means = None

        # Results
        self.bins = None
        self.raw_counts = None
        self.baseline_counts = None
        self.bin_to_score = None
        self.feature_to_score = None
        self.raw_stats = None
        self.summary_bins = None
        self.summary_counts = None
        self.best_param_idxs = None

        if schema is None:
            if self.schema_path is not None:
                logging.info("Loading schema from " + self.schema_path)
                self.schema = Schema.load(open(self.schema_path))
        else:
            self.schema = schema


    def save(self):
        logging.info("Saving job results to " + self.path)
        file = h5py.File(self.path, 'r+')

        # Saving feature ids is tricky because they are strings
        dt = h5py.special_dtype(vlen=str)
        ids = self.feature_ids
        self.feature_ids = file.create_dataset("feature_ids", (len(ids),), dt)
        for i, fid in enumerate(ids):
            self.feature_ids[i] = fid

        self.table = file.create_dataset("table", data=self.table)
        self.bins = file.create_dataset("bins", data=self.bins)
        self.raw_counts = file.create_dataset("raw_counts", data=self.raw_counts)
        self.baseline_counts = file.create_dataset("baseline_counts", data=self.baseline_counts)
        self.bin_to_score = file.create_dataset("bin_to_score", data=self.bin_to_score)
        self.feature_to_score = file.create_dataset("feature_to_score", data=self.feature_to_score)
        self.raw_stats = file.create_dataset("raw_stats", data=self.raw_stats)
        self.summary_bins = file.create_dataset("summary_bins", data=self.summary_bins)
        self.summary_counts = file.create_dataset("summary_counts", data=self.summary_counts)
        self.sample_indexes = file.create_dataset("sample_indexes", data=self.sample_indexes)
        self.best_param_idxs = file.create_dataset("best_param_idxs", data=self.best_param_idxs)


        schema_str = StringIO()
        self.schema.save(schema_str)
        file.attrs['schema'] = str(schema_str.getvalue())
        file.attrs['stat_name'] = self.stat
        file.attrs['min_conf'] = self.min_conf
        file.attrs['conf_levels'] = self.conf_levels
        file.attrs['num_bins'] = self.num_bins
        file.attrs['num_samples'] = self.num_samples
        file.attrs['sample_from'] = self.sample_from
        file.attrs['sample_method'] = self.sample_method
        file.attrs['full_model'] = str(self.full_model.expr)
        file.attrs['reduced_model'] = str(self.reduced_model.expr)

        file.close()

    def load(self):

        logging.info("Loading job results from " + self.path)
        file = h5py.File(self.path, 'r')

        self.table = file['table'][...]
        self.feature_ids = file['feature_ids'][...]
        self.bins = file['bins'][...]
        self.raw_counts = file['raw_counts'][...]
        self.baseline_counts = file['baseline_counts'][...]
        self.bin_to_score = file['bin_to_score'][...]
        self.feature_to_score = file['feature_to_score'][...]
        self.raw_stats = file['raw_stats'][...]
        self.summary_bins = file['summary_bins'][...]
        self.summary_counts = file['summary_counts'][...]
        self.sample_indexes = file['sample_indexes'][...]
        self.best_param_idxs = file['best_param_idxs'][...]

        schema_str = StringIO(file.attrs['schema'])
        self.schema = Schema.load(schema_str)
        self.stat_name = file.attrs['stat_name']
        self.min_conf = file.attrs['min_conf']
        self.conf_levels = file.attrs['conf_levels']
        self.num_bins = file.attrs['num_bins']
        self.num_samples = file.attrs['num_samples']
        self.sample_from = file.attrs['sample_from']
        self.sample_method = file.attrs['sample_method']
        self.full_model = Model(self.schema, file.attrs['full_model'])
        self.reduced_model = Model(self.schema, file.attrs['reduced_model'])
        
        file.close()


