"""Celery tasks for PADE workflow.

Everything in this module should be a celery task. Most of the work
should be delegated to function in pade.analysis. Each function should
take a job and possibly other parameters, compute something, modify
the job structure, and return it. This allows us to chain the tasks
together.

"""
from __future__ import absolute_import, print_function, division

import logging
import pade.analysis as an
import numpy as np
import h5py

from StringIO import StringIO
from pade.celery import celery
from pade.stat import (
    cumulative_hist, bins_uniform, confidence_scores, 
    assign_scores_to_features)
from pade.model import (
    Job, Settings, Results, Input, TableWithHeader, Summary, Schema)

def save_table(db, table, name):
    db.create_dataset(name, data=table.table)
    db[name].attrs['headers'] = table.header        

@celery.task
def copy_input(path, input_path, schema, settings, job_id):
    logging.info("Loading input for job from {0}".format(input_path))
    input = Input.from_raw_file(input_path, schema)
    logging.info("Saving input, settings, and schema to " + str(path))

    with h5py.File(path, 'w') as db:

        # Save the input object
        ids = input.feature_ids

        # Saving feature ids is tricky because they are strings
        dt = h5py.special_dtype(vlen=str)
        db.create_dataset("table", data=input.table)
        db.create_dataset("feature_ids", (len(ids),), dt)

        for i, fid in enumerate(ids):
            input.feature_ids[i] = fid
            db['feature_ids'][i] = fid

        # Save the settings object
        db.create_dataset("tuning_params", data=settings.tuning_params)
        db.attrs['job_id'] = job_id
        db.attrs['stat'] = settings.stat,
        db.attrs['glm_family'] = settings.glm_family,
        db.attrs['num_bins'] = settings.num_bins
        db.attrs['num_samples'] = settings.num_samples
        db.attrs['sample_from_residuals'] = settings.sample_from_residuals
        db.attrs['sample_with_replacement'] = settings.sample_with_replacement
        db.attrs['condition_variables'] = map(str, settings.condition_variables)
        db.attrs['block_variables'] = settings.block_variables
        db.attrs['summary_min_conf'] = settings.summary_min_conf
        db.attrs['summary_step_size'] = settings.summary_step_size
        db.attrs['equalize_means'] = settings.equalize_means

        # Save the schema object
        schema_str = StringIO()
        schema.save(schema_str)
        db.attrs['schema'] = str(schema_str.getvalue())

        if settings.equalize_means_ids is not None:
            db['equalize_means_ids'] = settings.equalize_means_ids

@celery.task
def load_sample_indexes(filename, job):
    job.results.sample_indexes = np.genfromtxt(args.sample_indexes, dtype=int)

@celery.task(name="Generate sample indexes")
def gen_sample_indexes(path):
    logging.info("Generating sample indexes for " + str(path))
    job = load_job(path)
    
    with h5py.File(path, 'r+') as db:
        indexes = an.new_sample_indexes(job)
        db.create_dataset("sample_indexes", data=indexes)

@celery.task
def compute_raw_stats(path):
    logging.info("Computing raw statistics")

    job = load_job(path)

    raw_stats    = job.get_stat_fn()(job.input.table)
    coeff_values = an.compute_coeffs(job)
    fold_change  = an.compute_fold_change(job)
    group_means  = an.compute_means(job)

    with h5py.File(path, 'r+') as db:
        db.create_dataset("raw_stats", data=raw_stats)
        save_table(db, group_means, 'group_means')
        save_table(db, fold_change, 'fold_change')
        save_table(db, coeff_values, 'coeff_values')
        
@celery.task
def choose_bins(path):
    logging.info("Choosing bins for discretized statistic space")
    job = load_job(path)
    bins = bins_uniform(job.settings.num_bins, job.results.raw_stats)
    with h5py.File(path, 'r+') as db:
        db.create_dataset("bins", data=bins)


@celery.task
def compute_mean_perm_count(path):
    logging.info("Computing mean permutation counts")
    job = load_job(path)
    bin_to_mean_perm_count = an.compute_mean_perm_count(job)
    with h5py.File(path, 'r+') as db:
        db.create_dataset("bin_to_mean_perm_count", data=bin_to_mean_perm_count)


@celery.task
def compute_conf_scores(path):
    logging.info("Computing confidence scores")
    job = load_job(path)
    raw  = job.results.raw_stats
    bins = job.results.bins
    
    unperm_counts = cumulative_hist(raw, bins)
    perm_counts   = job.results.bin_to_mean_perm_count
    bin_to_score  = confidence_scores(unperm_counts, perm_counts, np.shape(raw)[-1])
    feature_to_score = assign_scores_to_features(
        raw, bins, bin_to_score)

    with h5py.File(path, 'r+') as db:
        db.create_dataset("bin_to_unperm_count", data=unperm_counts)
        db.create_dataset("bin_to_score", data=bin_to_score)
        db.create_dataset("feature_to_score", data=feature_to_score)


@celery.task
def summarize_by_conf_level(path):
    logging.info("Summarizing counts by confidence level")
    job = load_job(path)
    summary = an.summary_by_conf_level(job)

    with h5py.File(path, 'r+') as db:
        grp = db.create_group('summary')
        grp['bins']            = summary.bins
        grp['best_param_idxs'] = summary.best_param_idxs
        grp['counts']          = summary.counts

        
@celery.task
def compute_orderings(path):

    logging.info("Computing orderings of features")
    job = load_job(path)
    original = np.arange(len(job.input.feature_ids))
    stats = job.results.feature_to_score[...]
    rev_stats = 0.0 - stats

    logging.info("  Computing ordering by score for each tuning param")
    by_score_original = np.zeros(np.shape(job.results.raw_stats), int)
    for i in range(len(job.settings.tuning_params)):
        by_score_original[i] = np.lexsort(
            (original, rev_stats[i]))

    order_by_score_original = by_score_original

    logging.info("  Computing ordering by fold change")
    by_foldchange_original = np.zeros(np.shape(job.results.fold_change.table), int)
    foldchange = job.results.fold_change.table[...]
    rev_foldchange = 0.0 - foldchange
    for i in range(len(job.results.fold_change.header)):
        keys = (original, rev_foldchange[..., i])

        by_foldchange_original[..., i] = np.lexsort(keys)

    order_by_foldchange_original = by_foldchange_original

    with h5py.File(path, 'r+') as db:
        orderings = db.create_group('orderings')
        orderings['by_score_original'] = order_by_score_original
        orderings['by_foldchange_original'] = order_by_foldchange_original        


def steps(settings, schema, infile_path, sample_indexes_path, path, job_id):

    do_copy_input = copy_input.si(path, infile_path, schema, settings, job_id)
    
    if sample_indexes_path is not None:
        make_sample_indexes = load_sample_indexes.si(path, os.path.abspath(sample_indexes_path))
    else:
        make_sample_indexes = gen_sample_indexes.si(path)

    return [

        # First we need to load the input table.
        do_copy_input,

        # Then construct (or load) a list of permutations of the indexes
        make_sample_indexes,

        # Then compute the raw statistics (f-test or other
        # differential expression stat, means, fold change, and
        # coefficients). We should be able to chunk this up. We would
        # then simply need to copy the chunk results into the master
        # job db.
        compute_raw_stats.si(path),

        # Choose bins for our histogram based on the values of the raw
        # stats. We would need to merge all of the above chunks first.
        choose_bins.si(path),

        # Then run the permutations and come up with cumulative
        # counts. This can be chunked. We would need to add another
        # step that merges the results together.
        compute_mean_perm_count.si(path),

        # Compare the unpermuted counts to the mean permuted counts to
        # come up with confidence scores.
        compute_conf_scores.si(path),

        # Produce a small summary table
        summarize_by_conf_level.si(path),

        # Now that we have all the stats, compute orderings using
        # different keys
        compute_orderings.si(path)]


def load_input(db):
    return Input(db['table'][...],
                 db['feature_ids'][...])

def load_job(path):

    with h5py.File(path, 'r') as db:
        return Job(
            job_id = db.attrs['job_id'],
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

    stat = db.attrs['stat']

    return Settings(
        stat = str(db.attrs['stat'][0]),
        glm_family = db.attrs['glm_family'][0],
        num_bins = db.attrs['num_bins'],
        num_samples = db.attrs['num_samples'],
        sample_from_residuals = db.attrs['sample_from_residuals'],
        sample_with_replacement = db.attrs['sample_with_replacement'],
        condition_variables = list(db.attrs['condition_variables']),
        block_variables = list(db.attrs['block_variables']),
        summary_min_conf = db.attrs['summary_min_conf'],
        summary_step_size = db.attrs['summary_step_size'],
        tuning_params = db['tuning_params'][...],
        equalize_means_ids = equalize_means_ids,
        equalize_means = db.attrs['equalize_means'])

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

