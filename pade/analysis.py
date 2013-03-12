import logging
import numpy as np
import pade.model
import scipy.stats
from collections import OrderedDict, namedtuple
from itertools import combinations

from pade.stat import (OneSampleDifferenceTTest, Ftest, MeansRatio, residuals)
from pade.confidence import bootstrap
from pade.model import TableWithHeader, Summary
from pade.layout import random_orderings, layout_is_paired

def predicted_values(job):
    """Return the values predicted by the reduced model.
    
    The return value has the same shape as the input table, with each
    cell containing the mean of all the cells in the same group, as
    defined by the reduced model.

    """
    data = job.input.table
    prediction = np.zeros_like(data)

    for grp in job.block_layout:
        means = np.mean(data[..., grp], axis=1)
        means = means.reshape(np.shape(means) + (1,))
        prediction[..., grp] = means
    return prediction



def summary_by_conf_level(job):
    """Summarize the counts by conf level.

    :param job:
      The pade.model.Job
    
    :return:

      A pade.model.Summary summarizing the results.

    """
    
    bins = np.arange(job.settings.min_conf, 1.0, job.settings.conf_interval)
    best_param_idxs = np.zeros(len(bins))
    counts          = np.zeros(len(bins))

    for i, conf in enumerate(bins):
        idxs = job.results.feature_to_score > conf
        best = np.argmax(np.sum(idxs, axis=1))
        best_param_idxs[i] = best
        counts[i]  = np.sum(idxs[best])

    return Summary(bins, best_param_idxs, counts)

def compute_coeffs(job):
    """Calculate the coefficients for the full model.

    :param job:
      The page.job.Job

    :return: 
      A TableWithHeader giving the coefficients for the linear model
      for each feature.

    """
    fitted = fit_model(job.full_model, job.input.table)
    names  = [assignment_name(a) for a in fitted.labels]    
    values = fitted.params
    return TableWithHeader(names, values)



def compute_fold_change(job):
    """Compute fold change.

    :param job:
      The pade.model.Job

    :return:
      A TableWithHeader giving the fold change for each non-baseline
      group for each feature.

    """
    
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

    return TableWithHeader(names, result)

def compute_means(job):
    """Compute the means for each group in the full model.
    
    :param job:
      The pade.model.Job

    :return:
      A TableWithHeader giving the mean for each of the blocking and
      condition variables.
    
    """
    factors = job.settings.block_variables + job.settings.condition_variables
    names = [assignment_name(a) 
             for a in job.schema.possible_assignments(factors)]
    values = get_group_means(job.schema, job.input.table, factors)
    return TableWithHeader(names, values)

def get_group_means(schema, data, factors):
    logging.debug("Getting group means for factors " + str(factors))
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
        constructor = OneSampleDifferenceTTest
    elif name == 'f_test':
        constructor = Ftest
    elif name == 'means_ratio':
        constructor = MeansRatio
    else:
        raise Exception("No statistic called " + str(job.settings.stat_name))

    if constructor == Ftest and layout_is_paired(job.block_layout):
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
            logging.debug("Bootstrapping using samples constructed from " +
                         "residuals, not using groups")
            layout = [ sorted(job.schema.sample_name_index.values()) ]
        else:
            logging.debug("Bootstrapping raw values, within groups defined by" + 
                         "'" + str(job.settings.block_variables) + "'")
            layout = job.block_layout
        return random_indexes(layout, R)

    else:
        logging.debug("Creating max of {0} random permutations".format(R))
        logging.debug("Condition layout is " + str(job.condition_layout))
        logging.debug("Block layout is " + str(job.block_layout))
        return list(random_orderings(job.condition_layout, job.block_layout, R))

    
def compute_mean_perm_count(job):

    table = job.input.table
    bins  = job.results.bins
    perms = job.results.sample_indexes
    stat_fn = get_stat_fn(job)

    if job.settings.sample_from_residuals:
        prediction = predicted_values(job)
        diffs      = table - prediction
        return bootstrap(
            prediction,
            stat_fn, 
            indexes=perms,
            residuals=diffs,
            bins=bins)

    else:
        # Shift all values in the data by the means of the groups from
        # the full model, so that the mean of each group is 0.
        if job.settings.equalize_means:
            shifted = residuals(table, job.full_layout)
            data = np.zeros_like(table)
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
                        data[i] = table[i]
                logging.debug("Equalized means for " + str(count - len(ids)) + " features")
                if len(ids) > 0:
                    logging.warn("There were " + str(len(ids)) + " feature " +
                                 "ids given that don't exist in the data: " +
                                 str(ids))

            return bootstrap(data, stat_fn, indexes=perms,bins=bins)

        else:
            return bootstrap(table, stat_fn, indexes=perms,bins=bins)


def assignment_name(a):

    if len(a) == 0:
        return "intercept"
    
    parts = ["{0}={1}".format(k, v) for k, v in a.items()]

    return ", ".join(parts)



DummyVarTable = namedtuple(
    "DummyVarTable",
    ["names", "rows"])

DummyVarAssignment = namedtuple(
    "DummyVarAssignment",
    ["factor_values",
     "bits",
     "indexes"])

FittedModel = namedtuple(
    "FittedModel",
    ["labels",
     "x",
     "y_indexes",
     "params"])

def dummy_vars(schema, factors=None, level=None):
    """
    level=0 is intercept only
    level=1 is intercept plus main effects
    level=2 is intercept, main effects, interactions between two variables
    ...
    level=n is intercept, main effects, interactions between n variables

    """ 
    factors = schema._check_factors(factors)

    if level is None:
        return dummy_vars(schema, factors, len(factors))

    if level == 0:
        names = ({},)
        rows = []
        for a in schema.possible_assignments(factors):
            rows.append(
                DummyVarAssignment(a.values(), (True,), schema.samples_with_assignments(a)))
        return DummyVarTable(names, rows)

    res = dummy_vars(schema, factors, level - 1)

    col_names = tuple(res.names)
    rows      = list(res.rows)

    # Get col names
    for interacting in combinations(factors, level):
        for a in schema.factor_combinations(interacting):
            if schema.has_baseline(dict(zip(interacting, a))):
                continue
            col_names += ({ interacting[i] : a[i] for i in range(len(interacting)) },)

    for i, dummy in enumerate(rows):
        (factor_values, bits, indexes) = dummy

        # For each subset of factors of size level
        for interacting in combinations(factors, level):

            my_vals = ()
            for j in range(len(factors)):
                if factors[j] in interacting:
                    my_vals += (factor_values[j],)

            # For each possible assignment of values to these factors
            for a in schema.factor_combinations(interacting):
                if schema.has_baseline(dict(zip(interacting, a))):
                    continue

                # Test if this row of the result table has all the
                # values in this assignment
                matches = my_vals == a
                bits = bits + (matches,)

        rows[i] = DummyVarAssignment(tuple(factor_values), bits, indexes)

    return DummyVarTable(col_names, rows)




def fit_model(model, data):

    logging.info("Computing coefficients using least squares for " +
             str(len(data)) + " rows")

    effect_level = 1 if model.expr.operator == '+' else len(model.expr.variables)

    dummies = dummy_vars(model.schema, level=effect_level, factors=model.expr.variables)

    x = []
    indexes = []

    for row in dummies.rows:
        for index in row.indexes:
            x.append(row.bits)
            indexes.append(model.schema.sample_name_index[index])

    x = np.array(x, bool)

    num_vars = np.size(x, axis=1)
    shape = np.array((len(data), num_vars))

    result = np.zeros(shape)

    for i, row in enumerate(data):
        y = row[indexes]
        (coeffs, residuals, rank, s) = np.linalg.lstsq(x, y)
        result[i] = coeffs

    return FittedModel(dummies.names, x, indexes, result)

