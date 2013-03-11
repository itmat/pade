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

@celery.task
def copy_input(job, filename):
    logging.info("Loading input for job from {0}".format(filename))
    job.input = pade.job.Input.from_raw_file(filename, job.schema)
    return job

@celery.task
def load_sample_indexes(filename, job):
    job.results.sample_indexes = np.genfromtxt(args.sample_indexes, dtype=int)
    return job

@celery.task(name="Generate sample indexes")
def gen_sample_indexes(job):
    logging.info("Generating sample indexes")
    job.results.sample_indexes = an.new_sample_indexes(job)
    return job


@celery.task
def compute_raw_stats(job):
    logging.info("Computing raw statistics")
    job.results.raw_stats    = an.get_stat_fn(job)(job.input.table)
    job.results.coeff_values = an.compute_coeffs(job)
    job.results.fold_change  = an.compute_fold_change(job)
    job.results.group_means  = an.compute_means(job)
    return job

@celery.task
def choose_bins(job):
    logging.info("Choosing bins for discretized statistic space")
    job.results.bins = pade.conf.bins_uniform(job.settings.num_bins, job.results.raw_stats)
    return job


@celery.task
def compute_mean_perm_count(job):
    logging.info("Computing mean permutation counts")
    job.results.bin_to_mean_perm_count = an.compute_mean_perm_count(job)
    return job

@celery.task
def compute_conf_scores(job):
    logging.info("Computing confidence scores")
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
def summarize_by_conf_level(job):
    logging.info("Summarizing counts by confidence level")
    job.summary = an.summary_by_conf_level(job)
    return job








@celery.task
def compute_orderings(job):
    logging.info("Computing orderings of features")
    original = np.arange(len(job.input.feature_ids))
    stats = job.results.feature_to_score[...]
    rev_stats = 0.0 - stats

    logging.info("  Computing ordering by score for each tuning param")
    by_score_original = np.zeros(np.shape(job.results.raw_stats), int)
    for i in range(len(job.settings.tuning_params)):
        by_score_original[i] = np.lexsort(
            (original, rev_stats[i]))

    job.results.order_by_score_original = by_score_original

    logging.info("  Computing ordering by fold change")
    by_foldchange_original = np.zeros(np.shape(job.results.fold_change.table), int)
    foldchange = job.results.fold_change.table[...]
    rev_foldchange = 0.0 - foldchange
    for i in range(len(job.results.fold_change.header)):
        keys = (original, rev_foldchange[..., i])

        by_foldchange_original[..., i] = np.lexsort(keys)

    job.results.order_by_foldchange_original = by_foldchange_original
    return job


@celery.task
def save_job(job, filename):
    logging.info("Saving job to {0}".format(filename))
    pade.job.save_job(filename, job)
    return job


def steps(settings, schema, infile_path, sample_indexes_path, output_path):

    copy_input = pade.tasks.copy_input.s(infile_path)
    
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
        pade.tasks.compute_orderings.s(),

        # and save the job.
        pade.tasks.save_job.s(output_path)]

    return steps
