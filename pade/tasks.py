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
def compute_coeffs(job):
    job.results.coeff_values = an.compute_coeffs(job)
    return job

@celery.task    
def compute_fold_change(job):
    job.results.fold_change = an.compute_fold_change(job)
    return job

@celery.task    
def compute_means(job):
    job.results.group_means = an.compute_means(job)
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
def run_sampling(job):

    stat = an.get_stat_fn(job)
    
    ###
    ### Compute raw stats
    ###
    
    logging.info("Computing {stat} statistics on raw data".format(stat=stat.name))
    raw_stats = stat(job.input.table)
    logging.debug("Shape of raw stats is " + str(np.shape(raw_stats)))

    logging.info("Creating {num_bins} bins based on values of raw stats".format(
            num_bins=job.settings.num_bins))
    job.results.bins = pade.conf.bins_uniform(job.settings.num_bins, raw_stats)

    if job.settings.sample_from_residuals:
        logging.info("Sampling from residuals")
        prediction = predicted_values(job)
        diffs      = job.input.table - prediction
        job.results.bin_to_mean_perm_count = pade.conf.bootstrap(
            prediction,
            stat, 
            indexes=job.results.sample_indexes,
            residuals=diffs,
            bins=job.results.bins)

    else:
        logging.info("Sampling from raw data")
        # Shift all values in the data by the means of the groups from
        # the full model, so that the mean of each group is 0.
        if job.settings.equalize_means:
            shifted = residuals(job.input.table, job.full_layout)
            data = np.zeros_like(job.input.table)
            if job.settings.equalize_means_ids is None:
                data = shifted
            else:
                ids = job.settings.equalize_means_ids
                count = len(ids)
                for i, fid in enumerate(job.input.feature_ids):
                    if fid in ids:
                        data[i] = shifted[i]
                        ids.remove(fid)
                    else:
                        data[i] = job.input.table[i]
                logging.info("Equalized means for " + str(count - len(ids)) + " features")
                if len(ids) > 0:
                    logging.warn("There were " + str(len(ids)) + " feature " +
                                 "ids given that don't exist in the data: " +
                                 str(ids))

            job.results.bin_to_mean_perm_count = pade.conf.bootstrap(
                data,
                stat, 
                indexes=job.results.sample_indexes,
                bins=job.results.bins)

        else:
            job.results.bin_to_mean_perm_count = pade.conf.bootstrap(
                job.input.table,
                stat, 
                indexes=job.results.sample_indexes,
                bins=job.results.bins)            

    logging.info("Done bootstrapping, now computing confidence scores")
    job.results.raw_stats    = raw_stats
    job.results.bin_to_unperm_count   = pade.conf.cumulative_hist(job.results.raw_stats, job.results.bins)
    job.results.bin_to_score = confidence_scores(
        job.results.bin_to_unperm_count, job.results.bin_to_mean_perm_count, np.shape(raw_stats)[-1])
    job.results.feature_to_score = assign_scores_to_features(
        job.results.raw_stats, job.results.bins, job.results.bin_to_score)

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
#    pade.job.save_job(filename, job)
    logging.info("Saved it")
    return job
