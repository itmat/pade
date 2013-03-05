import logging
import numpy as np
import pade.job
import scipy.stats
from collections import OrderedDict
from pade.common import assignment_name
import pade.stat as stat
from pade.layout import *


def summary_by_conf_level(job):
    """Summarize the counts by conf level.

    :param job:
      The pade.job.Job
    
    :return:

      A pade.job.Summary summarizing the results.

    """
    
    logging.info("Summarizing the results")

    
    bins = np.arange(job.settings.min_conf, 1.0, job.settings.conf_interval)
    best_param_idxs = np.zeros(len(bins))
    counts          = np.zeros(len(bins))

    for i, conf in enumerate(bins):
        idxs = job.results.feature_to_score > conf
        best = np.argmax(np.sum(idxs, axis=1))
        best_param_idxs[i] = best
        counts[i]  = np.sum(idxs[best])

    return pade.job.Summary(bins, best_param_idxs, counts)

def compute_coeffs(job):
    """Calculate the coefficients for the full model.

    :param job:
      The page.job.Job

    :return: 
      A TableWithHeader giving the coefficients for the linear model
      for each feature.

    """
    fitted = job.full_model.fit(job.input.table)
    names  = [assignment_name(a) for a in fitted.labels]    
    values = fitted.params
    return pade.job.TableWithHeader(names, values)

def compute_fold_change(job):
    """Compute fold change.

    :param job:
      The pade.job.Job

    :return:
      A TableWithHeader giving the fold change for each non-baseline
      group for each feature.

    """
    logging.info("Computing fold change")
    
    nuisance_factors = set(job.settings.block_variables)
    test_factors     = job.settings.condition_variables

    if len(test_factors) > 1:
        raise UsageException(
            """You can only have one condition variable. We will change this soon.""")

    nuisance_assignments = job.schema.possible_assignments(nuisance_factors)
    fold_changes = []
    names = []

    data = job.input.table
    get_means = lambda a: np.mean(data[:, job.schema.indexes_with_assignments(a)], axis=-1)

    alpha = scipy.stats.scoreatpercentile(job.input.table.flatten(), 1.0)

    for na in nuisance_assignments:
        test_assignments = job.schema.possible_assignments(test_factors)
        test_assignments = [OrderedDict(d.items() + na.items()) for d in test_assignments]
        layouts = [ job.schema.indexes_with_assignments(a) for a in test_assignments ]
        baseline_mean = get_means(test_assignments[0])
        for a in test_assignments[1:]:
            fold_changes.append((get_means(a) + alpha) / (baseline_mean + alpha))
            names.append(assignment_name(a))

    # Ignoring nuisance vars
    test_assignments = job.schema.possible_assignments(test_factors)
    baseline_mean = get_means(test_assignments[0])
    for a in test_assignments[1:]:
        fold_changes.append((get_means(a) + alpha) / (baseline_mean + alpha))
        names.append(assignment_name(a))
        
    num_features = len(data)
    num_groups = len(names)

    result = np.zeros((num_features, num_groups))
    for i in range(len(fold_changes)):
        result[..., i] = fold_changes[i]

    return pade.job.TableWithHeader(names, result)

def compute_means(job):
    """Compute the means for each group in the full model.
    
    :param job:
      The pade.job.Job

    :return:
      A TableWithHeader giving the mean for each of the blocking and
      condition variables.
    
    """
    factors = job.settings.block_variables + job.settings.condition_variables
    names = [assignment_name(a) 
             for a in job.schema.possible_assignments(factors)]
    values = get_group_means(job.schema, job.input.table, factors)
    return pade.job.TableWithHeader(names, values)

def get_group_means(schema, data, factors):
    logging.info("Getting group means for factors " + str(factors))
    assignments = schema.possible_assignments(factors=factors)

    num_features = len(data)
    num_groups = len(assignments)

    result = np.zeros((num_features, num_groups))

    for i, assignment in enumerate(assignments):
        indexes = schema.indexes_with_assignments(assignment)
        result[:, i] = np.mean(data[:, indexes], axis=1)

    return result

def get_stat_fn(job):
    """The statistic used for this job."""
    name = job.settings.stat_name

    if name == 'one_sample_t_test':
        constructor = pade.stat.OneSampleDifferenceTTest
    elif name == 'f_test':
        constructor = pade.stat.Ftest
    elif name == 'means_ratio':
        constructor = pade.stat.MeansRatio

    if constructor == pade.stat.Ftest and layout_is_paired(job.block_layout):
        raise UsageException(
"""I can't use the f-test with this data, because the reduced model
you specified has groups with only one sample. It seems like you have
a paired layout. If this is the case, please use the --paired option.
""")        

    return constructor(
        condition_layout=job.condition_layout,
        block_layout=job.block_layout,
        alphas=job.settings.tuning_params)

def new_sample_indexes(job):

    """Create array of sample indexes."""

    R  = job.settings.num_samples

    if job.settings.sample_with_replacement:
        if job.settings.sample_from_residuals:
            logging.info("Bootstrapping using samples constructed from " +
                         "residuals, not using groups")
            layout = [ sorted(job.schema.sample_name_index.values()) ]
        else:
            logging.info("Bootstrapping raw values, within groups defined by" + 
                         "'" + str(job.settings.block_variables) + "'")
            layout = job.block_layout
        logging.info("Layout is" + str(layout))
        return random_indexes(layout, R)

    else:
        logging.info("Creating max of {0} random permutations".format(R))
        return list(random_orderings(job.condition_layout, job.block_layout, R))

    
