#!/usr/bin/env python

"""The main program for page."""

# External imports

import argparse
import collections
import errno
import jinja2
import logging
import matplotlib.pyplot as plt
import numpy.ma as ma
import numpy as np
import h5py
from numpy.lib.recfunctions import append_fields
import os
import shutil
from itertools import combinations, product

from bisect import bisect

from page.common import *
from page.performance import *
from page.schema import *
from page.model import *
import page.stat
from page.stat import random_indexes, random_orderings, residuals, group_means

REAL_PATH = os.path.realpath(__file__)
RAW_VALUE_DTYPE = float
FEATURE_ID_DTYPE = 'S64'
DEFAULT_TUNING_PARAMS=np.array([0.001, 0.01, 0.1, 1, 3])

##############################################################################
###
### Exceptions
###

class UsageException(Exception):
    """Thrown when the user gave invalid parameters."""
    pass

##############################################################################
###
### Plotting and reporting
### 

StatDistPlot=namedtuple('StatDistPlot', ['tuning_param', 'filename'])

def plot_stat_dist(job, fdr):
    logging.info("Saving histograms of " + job.stat.name + " values")
    max_stat = np.max(fdr.raw_stats)
    for i, alpha in enumerate(DEFAULT_TUNING_PARAMS):
        filename = "images/raw_stats_" + str(i) + ".png"
        with figure(filename):
            plt.hist(fdr.raw_stats[i], log=False, bins=250)
            plt.title(job.stat.name + " distribution over features, $\\alpha = " + str(alpha) + "$")
            plt.xlabel(job.stat.name + " value")
            plt.ylabel("Features")
            plt.xlim(0, max_stat)
            yield StatDistPlot(alpha, filename)

def plot_conf_by_stat(fdr, filename='conf_by_stat',
                      extra=None):
    conf = fdr.bin_to_score
    bins = fdr.bins
    with figure(filename):
        plt.plot(bins[1:], conf, label='raw',)
        plt.plot(bins[1:], ensure_scores_increase(conf),
                 label='increasing')
        plt.xlabel('statistic')
        plt.ylabel('confidence')
        title = "Confidence level by statistic"
        if extra:
            title += extra
        plt.title(title)
        plt.semilogx()


def plot_counts_by_stat(fdr, filename='counts_by_stat', extra=None):
    raw_stats = fdr.raw_counts
    resampled_stats = fdr.baseline_counts
    bins = fdr.bins
    with figure(filename):
        plt.plot(bins[1:], raw_stats, label='raw data')
        plt.plot(bins[1:], resampled_stats, label='resampled')
        plt.loglog()
        title = 'Count of features by statistic'
        if extra is not None:
            title += extra
        plt.title(title)
        plt.xlabel('Statistic value')
        plt.ylabel('Features with statistic >= value')
        plt.legend()


def setup_css(env):
    """Copy the css from 996grid/code/css into its output location."""

    src = os.path.join(os.path.dirname(REAL_PATH), 'css')

    shutil.rmtree('css', True)
    shutil.copytree(src, 'css')

    with open('css/custom.css', 'w') as out:
        template = env.get_template('custom.css')
        out.write(template.render())


def ensure_scores_increase(scores):
    res = np.copy(scores)
    for i in range(1, len(res)):
        res[i] = max(res[i], res[i - 1])
    return res


@profiled
def predicted_values(job):
    """Return the values predicted by the reduced model.
    
    The return value has the same shape as the input table, with each
    cell containing the mean of all the cells in the same group, as
    defined by the reduced model.

    """
    data = job.source.table
    prediction = np.zeros_like(data)

    for grp in job.reduced_model.layout:
        means = np.mean(data[..., grp], axis=1)
        means = means.reshape(np.shape(means) + (1,))
        prediction[..., grp] = means
    return prediction




class ResultTable:

    def __init__(self,
                 group_names=None,
                 param_names=None,
                 means=None,
                 coeffs=None,
                 stats=None,
                 feature_ids=None,
                 scores=None,
                 min_score=None):
        self.means = means
        self.coeffs = coeffs
        self.group_names = group_names
        self.param_names = param_names
        self.stats = stats
        self.feature_ids = feature_ids
        self.scores = scores
        self.min_score = min_score
        

    def filter_by_score(self, min_score):
        idxs = self.scores > min_score
        best = np.argmax(np.sum(idxs, axis=1))
        idxs = idxs[best]
        stats = self.stats[best]
        scores = self.scores[best]
        return ResultTable(
            group_names=self.group_names,
            param_names=self.param_names,
            means=self.means[idxs],
            coeffs=self.coeffs[idxs],
            stats=stats[idxs],
            feature_ids=self.feature_ids[idxs],
            scores=scores[idxs],
            min_score=min_score)

    def __len__(self):
        return len(self.feature_ids)

    def pages(self, rows_per_page=100):
        for start in range(0, len(self), rows_per_page):
            size = min(rows_per_page, len(self) - start)
            end = start + size

            yield ResultTable(
                group_names=self.group_names,
                param_names=self.param_names,
                means=self.means[start : end],
                coeffs=self.coeffs[start : end],
                stats=self.stats[start : end],
                feature_ids=self.feature_ids[start : end],
                scores=self.scores[start : end])


def assignment_name(a):

    if len(a) == 0:
        return "intercept"
    
    parts = ["{0}={1}".format(k, v) for k, v in a.items()]

    return ", ".join(parts)
           
def do_run(args):
    job = run_job(args)
    job.save()

def do_report(args):

    job = Job(directory=args.job_directory,
              source=SourceData(args.source_directory))
    job.load()
    job.source.feature_ids
    job.source.table

    fitted = job.full_model.fit(job.source.table)
    means = get_group_means(job.source.schema, job.source.table)

    print "Stats are", job.raw_stats

    results = ResultTable(
        means=means,
        coeffs=fitted.params,
        group_names=[assignment_name(a) for a in job.source.schema.possible_assignments()],
        param_names=[assignment_name(a) for a in fitted.labels],
        feature_ids=np.array(job.source.feature_ids),
        stats=job.raw_stats,
        scores=job.feature_to_score)

    make_report(job, results, args.rows_per_page)

    with chdir(job.html_directory):
        print_profile(job)

 
@profiled
def run_job(args):

    source = SourceData(args.source_directory)

    job = Job(
        directory=args.job_directory,
        source=source,
        stat=args.stat,
        num_bins=args.num_bins,
        num_samples=args.num_samples,
        full_model=args.full_model,
        reduced_model=args.reduced_model,
        sample_from=args.sample_from,
        sample_method=args.sample_method,
        min_conf=args.min_conf,
        conf_levels=args.conf_levels)

    makedirs(job.directory)

    job.run()

    return job

def make_report(job, results, rows_per_page):

    with profiling('do_report: build report'):
        makedirs(job.html_directory)
        with chdir(job.html_directory):
            extra = "\nstat " + job.stat_name + ", sampling " + job.sample_from
            env = jinja2.Environment(loader=jinja2.PackageLoader('page'))
            setup_css(env)

            summary_bins = job.summary_bins
            summary_counts = np.zeros(len(summary_bins))

            template = env.get_template('conf_level.html')
            for level in range(len(job.summary_bins)):
                score=job.summary_bins[level]
                filtered = results.filter_by_score(score)
                summary_counts[level] = len(filtered)
                pages = list(filtered.pages(rows_per_page))
                for page_num, page in enumerate(pages):
                    with open('conf_level_{0}_page_{1}.html'.format(level, page_num), 'w') as out:
                        out.write(template.render(
                                conf_level=level,
                                min_score=score,
                                job=job,
                                results=page,
                                page_num=page_num,
                                num_pages=len(pages)))


            template = env.get_template('index.html')
            with open('index.html', 'w') as out:
                out.write(template.render(
                        job=job,
                        results=results,
                        summary_bins=job.summary_bins,
                        summary_counts=summary_counts))

#        with chdir(job.directory):
#            stat_dist_template = env.get_template('stat_dist.html')
#            with open('html/stat_dist.html', 'w') as out:
#                out.write(stat_dist_template.render(
#                        plots=plot_stat_dist(job, fdr)))

        print """
Summary of features by confidence level:

Confidence |   Num.
   Level   | Features
-----------+---------"""
        for i in range(len(summary_counts) - 1):
            print "{bin:10.0%} | {count:8d}".format(
                bin=summary_bins[i],
                count=int(summary_counts[i]))

        print """
The full report is available at {0}""".format(
            os.path.join(job.directory, "html/index.html"))

    save_text_output(job, results)
    return (job, results)

def save_text_output(job, results):
    with chdir(job.directory):
        (num_rows, num_cols) = job.source.table.shape
        num_cols += 2
        table = np.zeros((num_rows, num_cols))
        idxs = np.argmax(results.scores, axis=0)
        stats = [results.stats[idxs[i], i] for i in range(len(idxs))]
        scores = [results.scores[idxs[i], i] for i in range(len(idxs))]
        table = []

        # For each row in the data, add feature id, stat, score, group
        # means, and raw values.
        for i in range(len(job.source.table)):
            row = []
            row.append(job.source.feature_ids[i])
            row.append(results.stats[idxs[i], i])
            for j in range(len(DEFAULT_TUNING_PARAMS)):
                row.append(results.stats[j, i])
            row.append(results.scores[idxs[i], i])
            for j in range(len(DEFAULT_TUNING_PARAMS)):
                row.append(results.scores[j, i])
            row.extend(results.means[i])
            row.extend(results.coeffs[i])
            row.extend(job.source.table[i])
            table.append(tuple(row))
        schema = job.source.schema

        cols = []
        Col = namedtuple("Col", ['name', 'dtype', 'format'])
        def add_col(name, dtype, format):
            cols.append(Col(name, dtype, format))

        add_col(schema.feature_id_column_names[0], object, "%s")
        add_col('best_stat', float, "%f")
        for i, alpha in enumerate(DEFAULT_TUNING_PARAMS):
            add_col('stat_' + str(alpha), float, "%f")

        add_col('best_score', float, "%f")
        for i, alpha in enumerate(DEFAULT_TUNING_PARAMS):
            add_col('score_' + str(alpha), float, "%f")

        for name in results.group_names:
            add_col("mean: " + name, float, "%f")

        for name in results.param_names:
            add_col("param: " + name, float, "%f")

        for i, name in enumerate(schema.sample_column_names):
            add_col(name, float, "%f")
        
        dtype = [(c.name, c.dtype) for c in cols]

        table = np.array(table, dtype)
        with open("results.txt", "w") as out:
            out.write("\t".join(c.name for c in cols) + "\n")
            np.savetxt(out, table, 
                       fmt=[c.format for c in cols],
                       delimiter="\t")


@profiled
def assign_scores_to_features(stats, bins, scores):
    """Return an array that gives the confidence score for each feature.

    stats is an array giving the statistic value for each feature.

    bins is a monotonically increasing array which divides the
    statistic space up into ranges.

    scores is a monotonically increasing array of length (len(bins) -
    1) where scores[i] is the confidence level associated with
    statistics that fall in the range (bins[i-1], bins[i]).

    Returns an array that gives the confidence score for each feature.

    """
    logging.info("Assigning scores to features")
    logging.debug(("I have {num_stats} stats, {num_bins} bins, and " +
                  "{num_scores} scores").format(num_stats=np.shape(stats),
                                                num_bins=np.shape(bins),
                                                num_scores=np.shape(scores)))

    shape = np.shape(stats)
    res = np.zeros(shape)

    for idx in np.ndindex(shape):
        prefix = idx[:-1]
        stat = stats[idx]
        scores_idx = prefix + (bisect(bins[prefix], stat) - 1,)
        res[idx] = scores[scores_idx]
    logging.debug("Scores have shape {0}".format(np.shape(res)))
    return res


def adjust_num_diff(V0, R, num_ids):
    V = np.zeros((6,) + np.shape(V0))
    V[0] = V0
    for i in range(1, 6):
        V[i] = V[0] - V[0] / num_ids * (R - V[i - 1])
    return V[5]

@profiled
def confidence_scores(raw_counts, perm_counts, num_features):
    """Return confidence scores.
    
    """
    logging.info("Getting confidence scores for shape {shape} with {num_features} features".format(shape=np.shape(raw_counts),
                                                                                                   num_features=num_features))
    if np.shape(raw_counts) != np.shape(perm_counts):
        raise Exception(
            """raw_counts and perm_counts must have same shape.
               raw_counts is {raw} and perm_counts is {perm}""".format(
                raw=np.shape(raw_counts), perm=np.shape(perm_counts)))
    
    shape = np.shape(raw_counts)
    adjusted = np.zeros(shape)
    for idx in np.ndindex(shape[:-1]):
        adjusted[idx] = adjust_num_diff(perm_counts[idx], raw_counts[idx], num_features)

    res = (raw_counts - adjusted) / raw_counts

    for idx in np.ndindex(res.shape[:-1]):
        res[idx] = ensure_scores_increase(res[idx])

    return res


def get_group_means(schema, data):

    assignments = schema.possible_assignments()

    num_features = len(data)
    num_groups = len(assignments)

    result = np.zeros((num_features, num_groups))

    for i, assignment in enumerate(assignments):
        indexes = schema.indexes_with_assignments(assignment)
        result[:, i] = np.mean(data[:, indexes], axis=1)

    return result


##############################################################################
###
### Classes
###

class SourceData:

    DEFAULT_DIRECTORY = "page_source"

    def __init__(self, 
                 directory=DEFAULT_DIRECTORY,
                 schema=None,
                 
                 ):
        self.directory = directory

        # This will be populated lazily by the table and feature_ids
        # properties
        self._table = None
        self._feature_ids = None

        if schema is None:
            self.schema = Schema.load(open(self.schema_path))
        else:
            self.schema = schema


    def save_schema(self, mode):
        out = open(self.schema_path, mode)
        self.schema.save(out)

    @property
    def num_features(self):
        return len(self.feature_ids)

    @property
    def table_path(self):
        return os.path.join(self.directory, 'table.npy')

    @property
    def feature_ids_path(self):
        return os.path.join(self.directory, 'feature_ids.npy')

    @property
    def schema_path(self):
        """The path to the schema."""
        return os.path.join(self.directory, 'schema.yaml')

    @property
    def table(self):
        """The data table as a (sample x feature) ndarray."""
        if self._table is None:
            shape = (self.num_features,
                     len(self.schema.sample_column_names))
            self._table = np.memmap(
                self.table_path, 
                mode='r',
                shape=shape,
                dtype=RAW_VALUE_DTYPE)
        return self._table


    @property
    def feature_ids(self):
        """Array of the ids of features from my input file."""
        if self._feature_ids is None:
            self._feature_ids = np.memmap(
                self.feature_ids_path,
                mode='r',
                dtype=FEATURE_ID_DTYPE)

        return self._feature_ids

    def copy_table(self, raw_input_path):
        logging.info("Loading table from " + raw_input_path +
                     " to " + self.table_path + " and " + self.feature_ids_path)
        
        logging.info("Counting rows and columns in input file")
        with open(raw_input_path) as fh:

            headers = fh.next().rstrip().split("\t")
            num_cols = len(headers) - 1
            num_rows = 0
            for line in fh:
                num_rows += 1
        
        logging.info(
            "Input has {features} features and {samples} samples".format(
                features=num_rows,
                samples=num_cols))

        logging.info(
            "Creating raw data table")

        table = np.memmap(
            self.table_path, 
            mode='w+',
            dtype=RAW_VALUE_DTYPE,
            shape=(num_rows, num_cols))

        ids = np.memmap(
            self.feature_ids_path,
            mode='w+',
            dtype=FEATURE_ID_DTYPE,
            shape=(num_rows,))

        log_interval=int(num_rows / 10)

        with open(raw_input_path) as fh:

            headers = fh.next().rstrip().split("\t")

            for i, line in enumerate(fh):
                row = line.rstrip().split("\t")
                ids[i]   = row[0]
                table[i] = [float(x) for x in row[1:]]
                if (i % log_interval) == log_interval - 1:
                    logging.debug("Copied {0} rows".format(i + 1))


        del table
        del ids


class Job:

    DEFAULT_DIRECTORY = "page_job"

    def __init__(self, 
                 source=None,
                 directory=None, 
                 stat=None,
                 full_model=None,
                 reduced_model=None,
                 num_bins=None,
                 num_samples=None,
                 sample_from=None,
                 sample_method=None,
                 min_conf=None,
                 conf_levels=None):

        # Settings
        self.source = source
        self.directory = directory
        self.stat_name = stat
        self.min_conf = min_conf
        self.conf_levels = conf_levels
        self.num_bins = num_bins
        self.num_samples = num_samples
        self.sample_from = sample_from
        self.sample_method = sample_method
        self.full_model    = Model(self.source.schema, full_model)
        self.reduced_model = Model(self.source.schema, reduced_model)
        self._sample_indexes = None

        # Results
        self.bins = None
        self.raw_counts = None
        self.baseline_counts = None
        self.bin_to_score = None
        self.feature_to_score = None
        self.raw_stats = None
        self.summary_bins = None
        self.summary_counts = None


    def save(self):
        logging.info("Saving job results")
        file = h5py.File('job.hdf5')
        self.bins = file.create_dataset("bins", data=self.bins)
        self.raw_counts = file.create_dataset("raw_counts", data=self.raw_counts)
        self.baseline_counts = file.create_dataset("baseline_counts", data=self.baseline_counts)
        self.bin_to_score = file.create_dataset("bin_to_score", data=self.bin_to_score)
        self.feature_to_score = file.create_dataset("feature_to_score", data=self.feature_to_score)
        self.raw_stats = file.create_dataset("raw_stats", data=self.raw_stats)
        self.summary_bins = file.create_dataset("summary_bins", data=self.summary_bins)
        self.summary_counts = file.create_dataset("summary_counts", data=self.summary_counts)
        self._sample_indexes    = file.create_dataset("sample_indexes", data=self._sample_indexes)

        file.attrs['stat_name'] = self.stat_name
        file.attrs['min_conf'] = self.min_conf
        file.attrs['conf_levels'] = self.conf_levels
        file.attrs['num_bins'] = self.num_bins
        file.attrs['num_samples'] = self.num_samples
        file.attrs['sample_from'] = self.sample_from
        file.attrs['sample_method'] = self.sample_method
        file.attrs['full_model'] = str(self.full_model.expr)
        file.attrs['reduced_model'] = str(self.reduced_model.expr)

        file.close()

    def load(self, filename="job.hdf5"):
        logging.info("Loading job results")
        file = h5py.File('job.hdf5')
        bins = file['bins']
        self.raw_counts = file['raw_counts'][...]
        self.baseline_counts = file['baseline_counts'][...]
        self.bin_to_score = file['bin_to_score'][...]
        self.feature_to_score = file['feature_to_score'][...]
        self.raw_counts = file['raw_stats'][...]
        self.summary_bins = file['summary_bins'][...]
        self.summary_counts = file['summary_counts'][...]

        self.stat_name = file.attrs['stat_name']
        self.min_conf = file.attrs['min_conf']
        self.conf_levels = file.attrs['conf_levels']
        self.num_bins = file.attrs['num_bins']
        self.num_samples = file.attrs['num_samples']
        self.sample_from = file.attrs['sample_from']
        self.sample_method = file.attrs['sample_method']
        self.full_model = Model(self.source.schema, file.attrs['full_model'])
        self.reduced_model = Model(self.source.schema, file.attrs['reduced_model'])

        file.close()

    def save_sample_indexes(self, indexes):
        self._sample_indexes = indexes
        np.save(self.sample_indexes_path, indexes)

    def new_sample_indexes(self):

        method = (self.sample_method, self.sample_from)
        
        full = self.full_model
        reduced = self.reduced_model
        R = self.num_samples

        if method == ('perm', 'raw'):
            logging.info("Creating max of {0} random permutations".format(R))
            return list(random_orderings(full.layout, reduced.layout, R))

        elif method == ('boot', 'raw'):
            logging.info("Bootstrapping raw values, within groups defined by '" + 
                         str(reduced.expr) + "'")
            return random_indexes(reduced.layout, R)
        
        elif method == ('boot', 'residuals'):
            logging.info("Bootstrapping using samples constructed from residuals, not using groups")
            return random_indexes(
                [ sorted(self.schema.sample_name_index.values()) ], R)

        else:
            raise UsageException("Invalid sampling method")

    @property
    def sample_indexes(self):
        logging.debug("In Job.sample_indexes")
        if self._sample_indexes is None:
            path = self.sample_indexes_path

            logging.debug("_sample_indexes is none; initializing. Looking at " + 
                          path)
            if os.path.exists(path):
                logging.debug("Found sample indexes to load")
                self._sample_indexes = np.load(path)
            else:
                logging.debug("Sample indexes aren't created yet")
        return self._sample_indexes

    @property
    def html_directory(self):
        return os.path.join(self.directory, 'html')

    @property
    def images_directory(self):
        return os.path.join(self.directory, 'images')

    @property
    def sample_indexes_path(self):
        """Path to table of sample indexes"""
        return os.path.join(self.directory, 'sample_indexes.npy')

    @property
    def stat(self):
        """The statistic used for this job."""
        if self.stat_name == 'f':
            return page.stat.Ftest(
                layout_full=self.full_model.layout,
                layout_reduced=self.reduced_model.layout,
                alphas=DEFAULT_TUNING_PARAMS)
        elif self.stat_name == 'f_sqrt':
            return page.stat.FtestSqrt(
                layout_full=self.full_model.layout,
                layout_reduced=self.reduced_model.layout)
        elif self.stat_name == 't':
            return Ttest(alpha=1.0)

    @profiled
    def run(self):

        data = self.source.table
        stat = self.stat
        raw_stats = stat(data)

        logging.debug("Shape of raw stats is " + str(np.shape(raw_stats)))

        self.bins = page.stat.bins_uniform(self.num_bins, raw_stats)

        self.save_sample_indexes(self.new_sample_indexes())

        if self.sample_from == 'residuals':
            logging.info("Bootstrapping based on residuals")
            prediction = predicted_values(self)
            diffs      = data - prediction
            self.baseline_counts = page.stat.bootstrap(
                # data,
                prediction,
                stat, 
                indexes=self.sample_indexes,
                residuals=diffs,
                bins=self.bins)
        else:
            logging.info("Using raw data")
            # Shift all values in the data by the means of the groups from
            # the full model, so that the mean of each group is 0.
            # perm, raw, shifted: 113, 29
            # boot, raw, shifted: 103, 24
            # boot, residuals: 113, 32
            # perm, raw, unshifted: 113, 29
            # boot, raw, unshifted: 103, 24
            shifted = residuals(data, self.full_model.layout)

            self.baseline_counts = page.stat.bootstrap(
                shifted,
                stat, 
                indexes=self.sample_indexes,
                bins=self.bins)

        self.raw_stats    = raw_stats
        self.raw_counts   = page.stat.cumulative_hist(self.raw_stats, self.bins)
        self.bin_to_score = confidence_scores(
            self.raw_counts, self.baseline_counts, np.shape(raw_stats)[-1])
        self.feature_to_score = assign_scores_to_features(
            self.raw_stats, self.bins, self.bin_to_score)
        self.summary_bins = np.linspace(self.min_conf, 1.0, self.conf_levels)
        self.summary_counts = page.stat.cumulative_hist(
            self.feature_to_score, self.summary_bins)


class Report:

    DEFAULT_DIRECTORY = "page_report"

    def __init__(self, directory):
        self.directory = directory

def print_profile(job):

    walked = walk_profile()
    env = jinja2.Environment(loader=jinja2.PackageLoader('page'))
    setup_css(env)
    template = env.get_template('profile.html')
    with open('profile.html', 'w') as out:
        logging.info("Saving profile")
        out.write(template.render(profile=walked))

    fmt = []
    fmt += ["%d", "%d", "%s", "%f", "%f", "%f", "%f", "%f", "%f"]
    fmt += ["%d", "%d"]

    features = [ len(job.source.table) for row in walked ]
    samples  = [ len(job.source.table[0]) for row in walked ]

    print "row is", len(walked[0]), ", fmt is", len(fmt)
    walked = append_fields(walked, names='features', data=features)
    walked = append_fields(walked, names='samples', data=samples)


    with open('../profile.txt', 'w') as out:
        out.write("\t".join(walked.dtype.names) + "\n")
        np.savetxt(out, walked, fmt=fmt)


@profiled
def main():
    """Run pageseq."""

    args = get_arguments()

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
    else:
        console.setLevel(logging.ERROR)

    logging.getLogger('').addHandler(console)

    logging.info('Page starting')

    try:
        args.func(args)
    except UsageException as e:
        logging.fatal("Page exiting because of usage error")
        print fix_newlines(e.message)
        exit(1)
    
    logging.info('Page finishing')    


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

    return Schema(
        is_feature_id=is_feature_id,
        is_sample=is_sample,
        column_names=headers)


def init_job(infile, factors, directory, force=False):

    if isinstance(infile, str):
        infile = open(infile)
    schema = init_schema(infile=infile)    

    for name, values in factors.items():
        schema.add_factor(name, values)

    source = SourceData(directory, schema=schema)
    makedirs(source.directory)

    mode = 'w' if force else 'wx'
    try:
        source.save_schema(mode)
    except IOError as e:
        if e.errno == errno.EEXIST:
            raise UsageException("""\
                   The schema file \"{}\" already exists. If you want to
                   overwrite it, use the --force or -f argument.""".format(
                    source.schema_path))
        raise e

    source.copy_table(infile.name)

    return source

@profiled
def do_setup(args):
    source = init_job(
        infile=args.infile,
        directory=args.source_directory,
        factors={f : None for f in args.factor},
        force=args.force)

    print fix_newlines("""
I have generated a schema for your input file, with factors {factors}, and saved it to "{filename}". You should now edit that file to set the factors for each sample. The file contains instructions on how to edit it.
""").format(factors=source.schema.factors.keys(),
            filename=source.schema_path)

def add_reporting_args(p):
    grp = p.add_argument_group(
        title="reporting arguments")
    grp.add_argument(
        '--rows-per-page',
        type=int,
        default=100,
        help="Number of rows to display on each page of the report"),


def add_model_args(p):
    grp = p.add_argument_group(
        title="data model arguments",
        description="Use these options to specify the variables to use in the model.")
    
    grp.add_argument(
        '--full-model', '-M',

        help="""Specify the 'full' model. Required if there is more than one
class. For example, if you have factors 'batch' and 'treated', you could use
 'treated' or 'batch * treated'."""),

    grp.add_argument(
        '--reduced-model', '-m',
        help="""Specify the 'reduced' model. The format for the argument is the
 same as for --full-model."""),
    
    

def add_fdr_args(p):
    grp = p.add_argument_group(
        title="confidence estimation arguments",
        description="""These options control how we estimate the confidence levels. You can probably leave them unchanged, in which case I'll compute it using a permutation test with an f-test as the statistic, using a maximum of 1000 permutations.""")
    grp.add_argument(
        '--stat', '-s',
        choices=['f', 't', 'f_sqrt'],
        default='f',
        help="The statistic to use")

    grp.add_argument(
        '--num-samples', '-R',
        type=int,
        default=1000,
        help="The number of samples to use if bootstrapping, or the maximum number of permutations to use if doing permutation test.")

    grp.add_argument(
        '--sample-from',
        choices=['raw', 'residuals'],
        default='raw',
        help='Indicate whether to do bootstrapping based on samples from the raw data or sampled residuals')

    grp.add_argument(
        '--num-bins',
        type=int,
        default=1000,
        help="Number of bins to divide the statistic space into.")

    grp.add_argument(
        '--sample-method',
        default='perm',
        choices=['perm', 'boot'],
        help="""Whether to use a permutation test or bootstrapping to estimate confidence levels."""),

    grp.add_argument(
        '--min-conf',
        default=0.5,
        type=float,
        help="Smallest confidence level to report")

    grp.add_argument(
        '--conf-levels',
        default=11,
        type=int,
        help="Number of confidence levels to show")


def get_arguments():
    """Parse the command line options and return an argparse args
    object."""
    
    uberparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparsers = uberparser.add_subparsers(
        title='actions',
        description="""Normal usage is to run 'page.py setup ...', then manually edit the
schema.yaml file, then run 'page.py run ...'.""")

    # Set up "parent" parser, which contains some arguments used by all other parsers
    parents = argparse.ArgumentParser(add_help=False)
    parents.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Be verbose (print INFO level log messages)")
    parents.add_argument(
        '--debug', '-d', 
        action='store_true',
        help="Print debugging information")
    parents.add_argument(
        '--log',
        default="page.log",
        help="Location of log file")
    parents.add_argument(
        '--source-directory', '-S',
        default=SourceData.DEFAULT_DIRECTORY,
        help="The directory that contains the source data and schema"),
    parents.add_argument(
        '--job-directory', '-J',
        default=Job.DEFAULT_DIRECTORY,
        help="The directory that contains the intermediate results")
    parents.add_argument(
        '--report-directory',
        default=Report.DEFAULT_DIRECTORY,
        help="The directory to write the report to")

    # Create setup parser
    setup_parser = subparsers.add_parser(
        'setup',
        help="""Set up the job configuration. This reads the input file and
                outputs a YAML file that you then need to fill out in order to
                properly configure the job.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[parents])
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
        class.""")
    setup_parser.add_argument(
        '--force', '-f',
        action='store_true',
        help="""Overwrite any existing files""")
    setup_parser.set_defaults(func=do_setup)

    # Create "run" parser
    run_parser = subparsers.add_parser(
        'run',
        help="""Run the job.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[parents])

    add_model_args(run_parser)
    add_fdr_args(run_parser)
    run_parser.set_defaults(func=do_run)

    report_parser = subparsers.add_parser(
        'report',
        help="""Generate report""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[parents])
    add_reporting_args(report_parser)
    report_parser.set_defaults(func=do_report)

    return uberparser.parse_args()


if __name__ == '__main__':
    main()
