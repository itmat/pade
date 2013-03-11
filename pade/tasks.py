"""Celery tasks for PADE workflow.

Everything in this module should be a celery task. Most of the work
should be delegated to function in pade.analysis. Each function should
take a job and possibly other parameters, compute something, modify
the job structure, and return it. This allows us to chain the tasks
together.

"""
from __future__ import absolute_import

from pade.celery import celery

import logging
import pade.analysis as an
import numpy as np
import pade.conf
from pade.stat import *
from pade.conf import *
import h5py

@celery.task
def copy_input(db_path, input_path, schema, settings):
    logging.info("Loading input for job from {0}".format(input_path))
    input = pade.job.Input.from_raw_file(input_path, schema)

    logging.info("Saving input, settings, and schema to " + str(db_path))
    with h5py.File(db_path, 'w') as db:
        pade.job.save_input(input, db)
        pade.job.save_settings(settings, db)
        pade.job.save_schema(schema, db)
    return db_path

@celery.task
def load_sample_indexes(filename, job):
    job.results.sample_indexes = np.genfromtxt(args.sample_indexes, dtype=int)
    return job

@celery.task(name="Generate sample indexes")
def gen_sample_indexes(db_path):
    logging.info("Generating sample indexes for " + str(db_path))
    job = pade.job.load_job(db_path)
    
    with h5py.File(db_path, 'r+') as db:
        indexes = an.new_sample_indexes(job)
        db.create_dataset("sample_indexes", data=indexes)

    return db_path

@celery.task
def compute_raw_stats(path):
    logging.info("Computing raw statistics")

    job = pade.job.load_job(path)

    raw_stats    = an.get_stat_fn(job)(job.input.table)
    coeff_values = an.compute_coeffs(job)
    fold_change  = an.compute_fold_change(job)
    group_means  = an.compute_means(job)

    with h5py.File(path, 'r+') as db:
        db.create_dataset("raw_stats", data=raw_stats)
        pade.job.save_table(db, group_means, 'group_means')
        pade.job.save_table(db, fold_change, 'fold_change')
        pade.job.save_table(db, coeff_values, 'coeff_values')
        
    return path

@celery.task
def choose_bins(path):
    logging.info("Choosing bins for discretized statistic space")
    job = pade.job.load_job(path)
    bins = pade.conf.bins_uniform(job.settings.num_bins, job.results.raw_stats)
    with h5py.File(path, 'r+') as db:
        db.create_dataset("bins", data=bins)
    return path


@celery.task
def compute_mean_perm_count(path):
    logging.info("Computing mean permutation counts")
    job = pade.job.load_job(path)
    bin_to_mean_perm_count = an.compute_mean_perm_count(job)
    with h5py.File(path, 'r+') as db:
        db.create_dataset("bin_to_mean_perm_count", data=bin_to_mean_perm_count)
    return path

@celery.task
def compute_conf_scores(path):
    logging.info("Computing confidence scores")
    job = pade.job.load_job(path)
    raw  = job.results.raw_stats
    bins = job.results.bins
    
    unperm_counts = pade.conf.cumulative_hist(raw, bins)
    perm_counts   = job.results.bin_to_mean_perm_count
    bin_to_score  = confidence_scores(unperm_counts, perm_counts, np.shape(raw)[-1])
    feature_to_score = assign_scores_to_features(
        raw, bins, bin_to_score)

    with h5py.File(path, 'r+') as db:
        db.create_dataset("bin_to_unperm_count", data=unperm_counts)
        db.create_dataset("bin_to_score", data=bin_to_score)
        db.create_dataset("feature_to_score", data=feature_to_score)

    return path


@celery.task
def summarize_by_conf_level(path):
    logging.info("Summarizing counts by confidence level")
    job = pade.job.load_job(path)
    summary = an.summary_by_conf_level(job)

    with h5py.File(path, 'r+') as db:
        grp = db.create_group('summary')
        grp['bins']            = summary.bins
        grp['best_param_idxs'] = summary.best_param_idxs
        grp['counts']          = summary.counts

        
    return path

@celery.task
def compute_orderings(path):

    logging.info("Computing orderings of features")
    job = pade.job.load_job(path)
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

    return path


def steps(settings, schema, infile_path, sample_indexes_path, output_path):

    copy_input = pade.tasks.copy_input.s(infile_path, schema, settings)
    
    if sample_indexes_path is not None:
        make_sample_indexes = pade.tasks.load_sample_indexes.s(os.path.abspath(sample_indexes_path))
    else:
        make_sample_indexes = pade.tasks.gen_sample_indexes.s()

    steps = [

        # First we need to load the input table.
        copy_input,

        # Then construct (or load) a list of permutations of the indexes
        make_sample_indexes,

        # Then compute the raw statistics (f-test or other
        # differential expression stat, means, fold change, and
        # coefficients). We should be able to chunk this up. We would
        # then simply need to copy the chunk results into the master
        # job db.
        pade.tasks.compute_raw_stats.s(),

        # Choose bins for our histogram based on the values of the raw
        # stats. We would need to merge all of the above chunks first.
        pade.tasks.choose_bins.s(),

        # Then run the permutations and come up with cumulative
        # counts. This can be chunked. We would need to add another
        # step that merges the results together.
        pade.tasks.compute_mean_perm_count.s(),

        # Compare the unpermuted counts to the mean permuted counts to
        # come up with confidence scores.
        pade.tasks.compute_conf_scores.s(),

        # Produce a small summary table
        pade.tasks.summarize_by_conf_level.s(),

        # Now that we have all the stats, compute orderings using
        # different keys
        pade.tasks.compute_orderings.s()]

    return steps
