from __future__ import absolute_import, print_function, division

import matplotlib
matplotlib.use("Agg")
from matplotlib.ticker import FuncFormatter

import matplotlib.pyplot as plt
import numpy as np
import pade.tasks 
import logging

from flask import Blueprint, render_template, request, make_response, send_file, abort
from celery.result import AsyncResult
from bisect import bisect
from pade.stat import cumulative_hist, adjust_num_diff, GLMFStat
from StringIO import StringIO
from pade.metadb import JobMeta
from functools import wraps
from pade.analysis import assignment_name

bp = Blueprint(
    'job', __name__,
    template_folder='templates')

mdb = None

job_dbs = []

def job_context(f):
    @wraps(f)
    def decorated(*args, **kwargs):

        job_id = int(kwargs['job_id'])
        if mdb is not None:
            job_meta = mdb.job(job_id)
        else:
            job_meta = job_dbs[job_id]
        kwargs['job_meta'] = job_meta
        try:
            kwargs['job_db']   = pade.tasks.load_job(job_meta.path)
        except IOError as e:
            kwargs['job_db'] = None

        del kwargs['job_id']
        return f(*args, **kwargs)
    return decorated
    

def load_job(job_id):
    """Load the Job object with the given meta job id."""
    job_meta = get_job_meta(job_id)
    return job_db(job_meta)

def job_level_kwargs(job_meta):
    return {
        'job_id' : job_meta.obj_id,
        'job_name' : job_meta.name
        }

def all_job_metas():
    if mdb is None:
        return job_dbs
    else:
        return mdb.all_jobs()

@bp.route("/jobs")
def job_list():

    job_metas = all_job_metas()

    job_metas = sorted(job_metas, key=lambda f:f.obj_id, reverse=True)
    return render_template('jobs.html', jobs=job_metas, is_runner=(mdb is not None))

@bp.route("/jobs/<job_id>/")
@job_context
def job_details(job_meta, job_db):

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
    is_runner = mdb is not None

    kwargs = job_level_kwargs(job_meta)

    if job_meta.imported or task.status == 'SUCCESS':
        return render_template("job.html", job=job_db,
                               is_runner=is_runner,
                               **kwargs)

    else:
        return render_template(
            'job_status.html',
            is_runner=is_runner,
            status=task.status,
            **kwargs)


@bp.route("/jobs/<job_id>/conf_level/<conf_level>")
@job_context
def details(job_meta, job_db, conf_level):

    ### Process params
    conf_level = int(conf_level)
    alpha_idx = job_db.summary.best_param_idxs[conf_level]

    page_num = 0
    if 'page' in request.args:
        page_num = int(request.args.get('page'))
        
    scores = job_db.results.feature_to_score[alpha_idx]
    stats = job_db.results.raw_stats[alpha_idx]
    min_score = job_db.summary.bins[conf_level]

    rows_per_page = 50

    orig_idxs = np.arange(len(job_db.input.feature_ids))
    all_idxs = None
    order_name = request.args.get('order')
    if order_name is None:
        all_idxs      = np.arange(len(job_db.input.feature_ids))
    elif order_name == 'score_original':
        all_idxs = job_db.results.ordering_by_score_original[alpha_idx]
    elif order_name == 'foldchange_original':
        groupnum = int(request.args.get('groupnum'))
        all_idxs = job_db.results.ordering_by_foldchange_original[..., groupnum]

    filtered_idxs = all_idxs[scores[all_idxs] > min_score]
    start = page_num * rows_per_page
    end = start + rows_per_page
    idxs = filtered_idxs[ start : end ]

    score=job_db.summary.bins[conf_level]

    num_pages = int(np.ceil(float(len(filtered_idxs)) / float(rows_per_page)))

    kwargs = job_level_kwargs(job_meta)

    return render_template(
        "conf_level.html",
        num_pages=num_pages,
        page_num=page_num,

        conf_level=conf_level,
        min_score=score,
        indexes=idxs,
        group_names=job_db.results.group_means.header,
        coeff_names=job_db.results.coeff_values.header,
        fold_change_group_names=job_db.results.fold_change.header,
        stat_name=job_db.settings.stat,
        scores=scores[idxs],
        stats=stats[idxs],
        means=job_db.results.group_means.table[idxs],
        coeffs=job_db.results.coeff_values.table[idxs],
        feature_ids=job_db.input.feature_ids[idxs],
        fold_change=job_db.results.fold_change.table[idxs],
        **kwargs)

@bp.route("/jobs/<job_id>/features/<feature_num>")
@job_context
def feature(job_meta, job_db, feature_num):

    schema = job_db.schema
    feature_num = int(feature_num)
    factor_values = {
        s : { f : schema.get_factor(s, f) for f in schema.factors }
        for s in schema.sample_column_names }

    family = 'gaussian'
    if job_db.settings.glm_family != '':
        family = job_db.settings.glm_family 

    stat_fn = GLMFStat(condition_layout=job_db.condition_layout,
                       block_layout=job_db.condition_layout,
                       family=family)

    stats           = job_db.results.raw_stats[..., feature_num]
    params          = job_db.settings.tuning_params
    bins            = np.array([ bisect(job_db.results.bins[i], stats[i]) - 1 for i in range(len(params)) ])
    unperm_count    = np.array([ job_db.results.bin_to_unperm_count[i, bins[i]] for i in range(len(params))])
    mean_perm_count = np.array([ job_db.results.bin_to_mean_perm_count[i, bins[i]] for i in range(len(params))])
    adjusted        = np.array(adjust_num_diff(mean_perm_count, unperm_count, len(job_db.input.table)))
    new_scores      = (unperm_count - adjusted) / unperm_count
    max_stat        = job_db.results.bins[..., -2]
    measurements    = job_db.input.table[feature_num]
    fittedvalues    = stat_fn.fittedvalues(measurements)

    kwargs = job_level_kwargs(job_meta)

    return render_template(
        "feature.html",
        feature_num=feature_num,
        feature_id=job_db.input.feature_ids[feature_num],
        measurements=measurements,
        fittedvalues=fittedvalues,
        sample_names=job_db.schema.sample_column_names,
        factors=job_db.schema.factors,
        factor_values=factor_values,
        layout=job_db.full_layout,
        tuning_params=job_db.settings.tuning_params,
        stats=stats,
        bins=bins,
        job=job_db,
        num_bins=len(job_db.results.bins[0]),
        unperm_count=unperm_count,
        mean_perm_count=mean_perm_count,
        adjusted_perm_count=adjusted,
        max_stat=max_stat,
        scores=job_db.results.feature_to_score[..., feature_num],
        new_scores=new_scores,
        **kwargs)

@bp.route("/jobs/<job_id>/features/<feature_num>/measurement_scatter")
@job_context
def measurement_scatter(job_meta, job_db, feature_num):
    job = job_db
    feature_num = int(feature_num)
    schema = job.schema
    measurements = job.input.table[feature_num]

    plt.title('Measurements')
    plt.xlabel('Group')
    plt.ylabel('Measurement')

    assignments = schema.possible_assignments(job.full_variables)
    names = [assignment_name(a) for a in assignments]
    grps = [schema.indexes_with_assignments(a) for a in assignments]

    for i, a in enumerate(assignments):

        y = measurements[grps[i]]
        x = [i for j in y]
        plt.scatter(x, y)

    plt.xticks(np.arange(len(names)), 
               names,
               rotation=70
               )

    return figure_response()


@bp.route("/jobs/<job_id>/mean_vs_variance")
@job_context
def mean_vs_variance(job_meta, job_db):
    job   = job_db
    means = np.mean(job.input.table, axis=-1)
    var   = np.var(job.input.table, axis=-1)

    color = request.args.get('color')
    if color is None:
        color = 'score'


    if color == 'score':

        idxs = np.argmax(job_db.results.feature_to_score, axis=0)

        best_score = np.zeros((len(means)))

        for i in range(len(job.input.table)):
            idx = idxs[i]
            score = job.results.feature_to_score[idx, i]
            best_score[i] = score
        colors = best_score

    elif color == 'stat':
        colors = job.results.raw_stats[0]

    min_val = min(min(means), min(var))
    max_val = max(max(means), max(var))
    
    x = np.arange(min(max(means), max(var)))
    plt.plot(x, x)

#    plt.xlim((min_val, max_val))
#    plt.ylim((min_val, max_val))
    plt.title('Mean vs variance (' + job_meta.name + ")")
    plt.xlabel('Mean')
    plt.ylabel('Variance')
    plt.scatter(means, var, c=colors)
    plt.colorbar()
    return figure_response()


@bp.route("/jobs/<job_id>/features/<feature_num>/interaction_plot")
@job_context
def interaction_plot(job_meta, job_db, feature_num):
    job = job_db
    feature_num = int(feature_num)
    schema = job.schema
    measurements = job.input.table[feature_num]

    x_var = job.settings.condition_variables[0]
    if len(job.settings.block_variables) == 0:
        abort(404)

    series_var = job.settings.block_variables[0]

    assignments = schema.possible_assignments([series_var, x_var])

    ticks = schema.factor_values[x_var]

    plt.ylabel('Measurement')
    
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

        plt.errorbar(x=np.arange(len(ticks)), y=y, yerr=yerr, label=series_name)

    margin = 0.25

    plt.xlim(( 0 - margin, len(ticks) - 1 + margin))
    plt.xticks(np.arange(len(ticks)))
    fig, ax = plt.subplots()

    ax.set_xticklabels(ticks)
    plt.legend()
    plt.xlabel(x_var)
    return figure_response()


@bp.route("/jobs/<job_id>/features/<feature_num>/measurement_bars")
@job_context
def measurement_bars(job_meta, job_db, feature_num):
    job = job_db
    feature_num = int(feature_num)
    schema = job.schema
    measurements = job.input.table[feature_num]

    variables = job.full_variables
    if 'variable' in request.args:
        variables = [ request.args.get('variable') ]

    plt.title('Measurements for feature ' + job.input.feature_ids[feature_num] + " by " + ", ".join(variables))
    plt.ylabel('Measurement')
    
    assignments = schema.possible_assignments(variables)

    x = np.arange(len(assignments))
    width = 0.8

    y = []
    grps = [schema.indexes_with_assignments(a) for a in assignments]
    names = [", ".join(map(str, a.values())) for a in assignments]
    y = [ np.mean(measurements[g]) for g in grps]
    err = [ np.std(measurements[g]) for g in grps]
    plt.bar(x, y, yerr=err, color='y')
    plt.xticks(x+width/2., names, rotation=70)

    return figure_response()


@bp.route("/jobs/<job_id>/stat_dist")
@job_context
def stat_dist_plots_page(job_meta, job_db):
    semilogx = request.args.get('semilogx') == 'True'
    return render_template("stat_dist.html", 
                           job_id=job_meta.obj_id,
                           job=job_db,
                           semilogx=semilogx)

@bp.route("/jobs/<job_id>/feature_count_and_score_by_stat.html")
@job_context
def feature_count_and_score_by_stat(job_meta, job_db):
    semilogx = request.args.get('semilogx') == 'True'
    return render_template("feature_count_and_score_by_stat.html", 
                           job_id=job_meta.obj_id,
                           job=job_db,
                           semilogx=semilogx)

@bp.route("/jobs/<job_id>/confidence_dist")
def confidence_dist(job_id):
    return render_template("confidence_dist.html", 
                           job_id=job_id)

@bp.route("/jobs/<job_id>/stat_dist/<tuning_param>.png")
@job_context
def stat_dist_plot(job_meta, job_db, tuning_param):

    max_stat = np.max(job_db.results.raw_stats)
    tuning_param = int(tuning_param)
    title = job_db.settings.stat + " distribution over features, $\\alpha = " + str(tuning_param) + "$"

    plt.title(title)
    plt.xlabel(job_db.settings.stat + " value")
    plt.ylabel("Features")
    plt.xlim((0, max_stat))
    plt.hist(job_db.results.raw_stats[tuning_param], log=False, bins=250)
    return figure_response()

@bp.route("/jobs/<job_id>/bin_to_score.png")
@job_context
def bin_to_score_plot(job_meta, job_db):
    data = job_db.results.bin_to_score
    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        title="Confidence by stat value value",
        xlabel="Statistic value",
        ylabel="Confidence")

    for i, param in enumerate(job_db.settings.tuning_params):
        plt.plot(job_db.results.bins[i, :-1], data[i], label=str(param))

    if request.args.get('semilogx') == 'True':
        plt.semilogx(base=10)
    plt.legend(loc='lower right')

    return figure_response()

@bp.route("/jobs/<job_id>/bin_to_features.png")
@job_context
def bin_to_features_plot(job_meta, job_db):

    params = job_db.settings.tuning_params
    if 'tuning_param_idx' in request.args:
        params = [ params[int(request.args.get('tuning_param_idx'))] ]

    plt.title('Features count by statistic value')
    plt.xlabel('Statistic value')
    plt.ylabel('Features')

    for i, param in enumerate(params):
        plt.plot(job_db.results.bins[i, :-1], job_db.results.bin_to_mean_perm_count[i], '--', label=str(param) + " permuted")
        plt.plot(job_db.results.bins[i, :-1], job_db.results.bin_to_unperm_count[i], label=str(param) + " unpermuted")
    if request.args.get('semilogx') == 'True':
        plt.semilogx(base=10)
    plt.legend(loc='upper right')
    return figure_response()

def max_features_found():
    counts = []
    for jm in all_job_metas():
        jdb = pade.tasks.load_job(jm.path)
        counts.append(max(jdb.summary.counts))
    return max(counts)

@bp.route("/jobs/<job_id>/conf_dist")
@job_context
def conf_dist_plot(job_meta, job_db):
    max_count = max_features_found()

    plt.title("Feature count by confidence score")
    plt.xlabel("Confidence score")
    plt.ylabel("Features")
    plt.plot(job_db.summary.bins, job_db.summary.counts)
    plt.ylim((0, max_count))
    return figure_response()


@bp.route("/conf_dist")
def all_conf_dist_plot():

    plt.title("Feature count by confidence score")
    plt.xlabel("Confidence score")
    plt.ylabel("Features")

    for jm in all_job_metas():
        jdb = pade.tasks.load_job(jm.path)
        plt.plot(jdb.summary.bins, jdb.summary.counts, label=jm.name)
    plt.legend(loc='upper right')
    return figure_response()
    

@bp.route("/jobs/<job_id>/score_dist_for_tuning_params.png")
@job_context
def score_dist_by_tuning_param(job_meta, job_db):

    plt.title('Features by confidence score')
    plt.xlabel('Confidence')
    plt.ylabel('Features')

    lines = []
    labels = []

    params = job_db.settings.tuning_params
    if 'tuning_param_idx' in request.args:
        params = [ int(request.args.get('tuning_param_idx')) ]

    for i, alpha in enumerate(params):
        bins = np.arange(0.5, 1.0, 0.01)
        hist = cumulative_hist(job_db.results.feature_to_score[i], bins)
        lines.append(plt.plot(bins[:-1], hist, label=str(alpha)))
        labels.append(str(alpha))
    plt.legend(loc='upper right')

    return figure_response()


def figure_response():
    """Turns a matplotlib figure into an HTTP response."""
    png_output = StringIO()
    plt.savefig(png_output)
    plt.clf()
    response = make_response(png_output.getvalue())
    response.headers['Content-Type'] = 'image/png'
    return response

@bp.route("/jobs/<job_id>/result_db")
@job_context
def result_db(job_meta, job_db):
    return send_file(job_meta.path, as_attachment=True)


