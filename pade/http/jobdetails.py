from __future__ import absolute_import, print_function, division

import matplotlib
matplotlib.use("Agg")
from matplotlib.ticker import FuncFormatter

import matplotlib.pyplot as plt
import numpy as np
import pade.tasks 

from flask import Blueprint, render_template, request, make_response
from celery.result import AsyncResult
from bisect import bisect
from pade.stat import cumulative_hist, adjust_num_diff
from StringIO import StringIO

bp = Blueprint(
    'job', __name__,
    template_folder='templates')

def load_job(job_id):
    """Load the Job object with the given meta job id."""
    job_meta = mdb.job(job_id)
    return pade.tasks.load_job(job_meta.path)

@bp.route("/")
def job_details(job_id):

    job_meta = mdb.job(job_id)

    if job_meta.imported:
        task = None
    else:
        task_ids = mdb.get_task_ids(job_meta)
        tasks = [ AsyncResult(x) for x in task_ids ]

        if len(tasks) != 1:
            raise Exception("I got " + str(len(tasks)) +
                            " tasks for job " + 
                            str(job_id) + "; this should never happen")
        task = tasks[0]

    if job_meta.imported or task.status == 'SUCCESS':
        job = load_job(job_id)
        return render_template("job.html", job_id=job.job_id, job=job)

    else:
        return render_template(
            'job_status.html',
            job_id=job_id,
            status=task.status)


@bp.route("/conf_level/<conf_level>")
def details(job_id, conf_level):
    job = load_job(job_id)

    ### Process params
    conf_level = int(conf_level)
    alpha_idx = job.summary.best_param_idxs[conf_level]

    page_num = 0
    if 'page' in request.args:
        page_num = int(request.args.get('page'))

        
    scores = job.results.feature_to_score[alpha_idx]
    stats = job.results.raw_stats[alpha_idx]
    min_score = job.summary.bins[conf_level]

    rows_per_page = 50

    orig_idxs = np.arange(len(job.input.feature_ids))
    all_idxs = None
    order_name = request.args.get('order')
    if order_name is None:
        all_idxs      = np.arange(len(job.input.feature_ids))
    elif order_name == 'score_original':
        all_idxs = job.results.ordering_by_score_original[alpha_idx]
    elif order_name == 'foldchange_original':
        groupnum = int(request.args.get('groupnum'))
        all_idxs = job.results.ordering_by_foldchange_original[..., groupnum]

    filtered_idxs = all_idxs[scores[all_idxs] > min_score]
    start = page_num * rows_per_page
    end = start + rows_per_page
    idxs = filtered_idxs[ start : end ]

    score=job.summary.bins[conf_level]

    num_pages = int(np.ceil(float(len(filtered_idxs)) / float(rows_per_page)))

    return render_template(
        "conf_level.html",
        job=job,
        job_id=job.job_id,
        num_pages=num_pages,
        conf_level=conf_level,
        min_score=score,
        indexes=idxs,
        group_names=job.results.group_means.header,
        coeff_names=job.results.coeff_values.header,
        fold_change_group_names=job.results.fold_change.header,
        stat_name=job.settings.stat_class.NAME,
        scores=scores[idxs],
        stats=scores[idxs],
        means=job.results.group_means.table[idxs],
        coeffs=job.results.coeff_values.table[idxs],
        feature_ids=job.input.feature_ids[idxs],
        fold_change=job.results.fold_change.table[idxs],
        page_num=page_num)

@bp.route("/features/<feature_num>")
def feature(job_id, feature_num):
    job = load_job(job_id)
    schema = job.schema
    feature_num = int(feature_num)
    factor_values = {
        s : { f : schema.get_factor(s, f) for f in schema.factors }
        for s in schema.sample_column_names }

    stats=job.results.raw_stats[..., feature_num]

    params = job.settings.tuning_params

    bins = np.array([ bisect(job.results.bins[i], stats[i]) - 1 for i in range(len(params)) ])
    unperm_count=np.array([ job.results.bin_to_unperm_count[i, bins[i]] for i in range(len(params))])
    mean_perm_count=np.array([ job.results.bin_to_mean_perm_count[i, bins[i]] for i in range(len(params))])

    adjusted=np.array(adjust_num_diff(mean_perm_count, unperm_count, len(job.input.table)))

    new_scores = (unperm_count - adjusted) / unperm_count

    max_stat = job.results.bins[..., -2]

    return render_template(
        "feature.html",
        feature_num=feature_num,
        feature_id=job.input.feature_ids[feature_num],
        measurements=job.input.table[feature_num],
        sample_names=job.schema.sample_column_names,
        factors=job.schema.factors,
        factor_values=factor_values,
        layout=job.full_layout,
        tuning_params=job.settings.tuning_params,
        stats=stats,
        bins=bins,
        job=job,
        job_id=job.job_id,
        num_bins=len(job.results.bins[0]),
        unperm_count=unperm_count,
        mean_perm_count=mean_perm_count,
        adjusted_perm_count=adjusted,
        max_stat=max_stat,
        scores=job.results.feature_to_score[..., feature_num],
        new_scores=new_scores
        )

@bp.route("/features/<feature_num>/measurement_scatter")
def measurement_scatter(job_id, feature_num):
    job = load_job(job_id)    
    feature_num = int(feature_num)


    schema = job.schema
    measurements = job.input.table[feature_num]
    
    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        title='Measurements',
        xlabel='Group',
        ylabel='Measurement')

    assignments = schema.possible_assignments(job.full_variables)
    names = [assignment_name(a) for a in assignments]
    grps = [schema.indexes_with_assignments(a) for a in assignments]

    for i, a in enumerate(assignments):

        y = measurements[grps[i]]
        x = [i for j in y]
        ax.scatter(x, y)

    plt.xticks(np.arange(len(names)), 
               names,
               rotation=70
               )

    ax.legend(loc='upper_right')
    return figure_response(fig)


@bp.route("/mean_vs_std")
def mean_vs_std(job_id):
    job = load_job(job_id)
    means = np.mean(job.input.table, axis=-1)
    std   = np.std(job.input.table, axis=-1)
    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        title='Mean vs standard deviation',
        xlabel='Mean',
        ylabel='Standard deviation')

    ax.scatter(means, std)
    return figure_response(fig)


@bp.route("/features/<feature_num>/interaction_plot")
def interaction_plot(job_id, feature_num):
    job = load_job(job_id)
    feature_num = int(feature_num)
    schema = job.schema
    measurements = job.input.table[feature_num]

    x_var = job.settings.condition_variables[0]
    series_var = job.settings.block_variables[0]
    assignments = schema.possible_assignments([series_var, x_var])

    ticks = schema.factor_values[x_var]

    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        ylabel='Measurement')
    
    for series_name in schema.factor_values[series_var]:
        y = []
        yerr = []
        for i, tick in enumerate(ticks):
            a = { series_var : series_name,
                  x_var : tick }
            idxs = schema.indexes_with_assignments(a)
            values = measurements[idxs]
            y.append(np.mean(values))
            yerr.append(np.std(values))

        ax.errorbar(x=np.arange(len(ticks)), y=y, yerr=yerr, label=series_name)

    margin = 0.25

    ax.set_xlim(( 0 - margin, len(ticks) - 1 + margin))
    ax.set_xticks(np.arange(len(ticks)))
    ax.set_xticklabels(ticks)

    ax.legend()
    ax.xlabel = x_var
    return figure_response(fig)


@bp.route("/features/<feature_num>/measurement_bars")
def measurement_bars(job_id, feature_num):
    job = load_job(job_id)
    feature_num = int(feature_num)
    schema = job.schema
    measurements = job.input.table[feature_num]

    variables = job.full_variables
    if 'variable' in request.args:
        variables = [ request.args.get('variable') ]

    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        title='Measurements for feature ' + job.input.feature_ids[feature_num] + " by " + ", ".join(variables),
        ylabel='Measurement')
    
    assignments = schema.possible_assignments(variables)

    x = np.arange(len(assignments))
    width = 0.8

    y = []
    grps = [schema.indexes_with_assignments(a) for a in assignments]
    names = [", ".join(a.values()) for a in assignments]
    y = [ np.mean(measurements[g]) for g in grps]
    err = [ np.std(measurements[g]) for g in grps]
    ax.bar(x, y, yerr=err, color='y')
    plt.xticks(x+width/2., names, rotation=70)

    return figure_response(fig)




@bp.route("/stat_dist")
def stat_dist_plots_page(job_id):
    job = load_job(job_id)
    semilogx = request.args.get('semilogx') == 'True'
    return render_template("stat_dist.html", 
                           job_id=job_id,
                           job=job,
                           semilogx=semilogx)

@bp.route("/feature_count_and_score_by_stat.html")
def feature_count_and_score_by_stat(job_id):
    job = load_job(job_id)
    semilogx = request.args.get('semilogx') == 'True'
    return render_template("feature_count_and_score_by_stat.html", 
                           job_id=job_id,
                           job=job,
                           semilogx=semilogx)

@bp.route("/confidence_dist")
def confidence_dist(job_id):
    return render_template("confidence_dist.html", 
                           job_id=job_id)

@bp.route("/stat_dist/<tuning_param>.png")
def stat_dist_plot(job_id, tuning_param):
    job = load_job(job_id)
    max_stat = np.max(job.results.raw_stats)
    tuning_param = int(tuning_param)
    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        title=job.settings.stat_class + " distribution over features, $\\alpha = " + str(tuning_param) + "$",
        xlabel=job.settings.stat_class.NAME + " value",
        ylabel="Features",
        xlim=(0, max_stat))
    plt.hist(job.results.raw_stats[tuning_param], log=False, bins=250)
    return figure_response(fig)

@bp.route("/bin_to_score.png")
def bin_to_score_plot(job_id):
    job = load_job(job_id)
    data = job.results.bin_to_score
    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        title="Confidence by stat value value",
        xlabel="Statistic value",
        ylabel="Confidence")

    for i, param in enumerate(job.settings.tuning_params):
        ax.plot(job.results.bins[i, :-1], data[i], label=str(param))

    if request.args.get('semilogx') == 'True':
        ax.semilogx(base=10)
    ax.legend(loc='lower right')

    return figure_response(fig)

@bp.route("/bin_to_features.png")
def bin_to_features_plot(job_id):
    job = load_job(job_id)
    params = job.settings.tuning_params
    if 'tuning_param_idx' in request.args:
        params = [ params[int(request.args.get('tuning_param_idx'))] ]

    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        title='Features count by statistic value',
        xlabel='Statistic value',
        ylabel='Features')
    job = job
    for i, param in enumerate(params):
        ax.plot(job.results.bins[i, :-1], job.results.bin_to_mean_perm_count[i], '--', label=str(param) + " permuted")
        ax.plot(job.results.bins[i, :-1], job.results.bin_to_unperm_count[i], label=str(param) + " unpermuted")
    if request.args.get('semilogx') == 'True':
        ax.semilogx(base=10)
    ax.legend(loc='upper right')
    return figure_response(fig)

@bp.route("/conf_dist")
def conf_dist_plot(job_id):
    job = load_job(job_id)
    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        title="Feature count by confidence score",
        xlabel="Confidence score",
        ylabel="Features")
    ax.plot(job.summary.bins, job.summary.counts)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos=0: "{:.0f}%".format(x * 100)))
    return figure_response(fig)
    

@bp.route("/score_dist_for_tuning_params.png")
def score_dist_by_tuning_param(job_id):
    job = load_job(job_id)
    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        title='Features by confidence score',
        xlabel='Confidence',
        ylabel='Features')

    lines = []
    labels = []

    params = job.settings.tuning_params
    if 'tuning_param_idx' in request.args:
        params = [ int(request.args.get('tuning_param_idx')) ]

    for i, alpha in enumerate(params):
        bins = np.arange(0.5, 1.0, 0.01)
        hist = cumulative_hist(job.results.feature_to_score[i], bins)
        lines.append(ax.plot(bins[:-1], hist, label=str(alpha)))
        labels.append(str(alpha))
    ax.legend(loc='upper right')

    return figure_response(fig)


def figure_response(fig):
    """Turns a matplotlib figure into an HTTP response."""
    png_output = StringIO()
    fig.savefig(png_output)
    response = make_response(png_output.getvalue())
    response.headers['Content-Type'] = 'image/png'
    return response

