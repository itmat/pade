from __future__ import absolute_import

from pade.celery import celery

import logging
import pade.analysis as an
import numpy as np
import pade.conf
from pade.stat import *
from pade.conf import *

@celery.task
def summarize_by_conf_level(job):
    job.summary = an.summary_by_conf_level(job)
    return job

@celery.task
def compute_orderings(job):

    original = np.arange(len(job.input.feature_ids))
    stats = job.results.feature_to_score[...]
    rev_stats = 0.0 - stats

    by_score_original = np.zeros(np.shape(job.results.raw_stats), int)
    for i in range(len(job.settings.tuning_params)):
        by_score_original[i] = np.lexsort(
            (original, rev_stats[i]))

    job.results.order_by_score_original = by_score_original

    by_foldchange_original = np.zeros(np.shape(job.results.fold_change.table), int)
    foldchange = job.results.fold_change.table[...]
    rev_foldchange = 0.0 - foldchange
    for i in range(len(job.results.fold_change.header)):
        keys = (original, rev_foldchange[..., i])

        by_foldchange_original[..., i] = np.lexsort(keys)

    job.results.order_by_foldchange_original = by_foldchange_original
    return job

@celery.task
def compute_raw_stats(job):
    job.results.raw_stats    = an.get_stat_fn(job)(job.input.table)
    job.results.coeff_values = an.compute_coeffs(job)
    job.results.fold_change  = an.compute_fold_change(job)
    job.results.group_means  = an.compute_means(job)
    return job

@celery.task
def choose_bins(job):
    job.results.bins = pade.conf.bins_uniform(job.settings.num_bins, job.results.raw_stats)
    return job

@celery.task
def compute_mean_perm_count(job):
    job.results.bin_to_mean_perm_count = an.compute_mean_perm_count(job)
    return job

@celery.task
def compute_conf_scores(job):
    raw  = job.results.raw_stats
    bins = job.results.bins
    unperm_counts = pade.conf.cumulative_hist(raw, bins)
    perm_counts   = job.results.bin_to_mean_perm_count
    bin_to_score  = confidence_scores(unperm_counts, perm_counts, np.shape(raw)[-1])

    job.results.bin_to_unperm_count = unperm_counts
    job.results.bin_to_score = bin_to_score
    job.results.feature_to_score = assign_scores_to_features(
        raw, bins, bin_to_score)

    return job

@celery.task
def copy_input(job, filename):
    job.input = pade.job.Input.from_raw_file(filename, job.schema)
    return job

@celery.task
def load_sample_indexes(filename, job):
    job.results.sample_indexes = np.genfromtxt(args.sample_indexes, dtype=int)
    return job

@celery.task
def gen_sample_indexes(job):
    job.results.sample_indexes = an.new_sample_indexes(job)
    return job

@celery.task
def save_job(job, filename):
    logging.info("Job is " + str(job))
    logging.info("Filename is " + str(filename))
    pade.job.save_job(filename, job)
    logging.info("Saved it")
    return job
