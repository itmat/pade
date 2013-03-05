# TODO:
#  Pre-generate histograms of stat distributions

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import argparse
import logging 
import StringIO
import pade.conf
from bisect import bisect
from flask import Flask, render_template, make_response, request
from pade.common import *
from pade.job import Job
from pade.conf import cumulative_hist


class PadeApp(Flask):

    def __init__(self):
        super(PadeApp, self).__init__(__name__)
        self.job = None

app = PadeApp()

def figure_response(fig):
    png_output = StringIO.StringIO()
    fig.savefig(png_output)
    response = make_response(png_output.getvalue())
    response.headers['Content-Type'] = 'image/png'
    return response

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/job")
def job():
    logging.info("Getting index")
    return render_template("job.html", job=app.job)

@app.route("/measurement_scatter/<feature_num>")
def measurement_scatter(feature_num):
    
    feature_num = int(feature_num)

    job = app.job
    schema = job.schema
    model = job.full_model
    measurements = job.input.table[feature_num]
    
    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        title='Measurements',
        xlabel='Group',
        ylabel='Measurement')

    assignments = schema.possible_assignments(model.expr.variables)
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

@app.route("/mean_vs_std")
def mean_vs_std():
    job = app.job
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

@app.route("/features/<feature_num>/measurement_bars")
def measurement_bars(feature_num):
    
    feature_num = int(feature_num)

    job = app.job
    schema = job.schema
    model = job.full_model
    measurements = job.input.table[feature_num]

    variables = model.expr.variables
    if 'variable' in request.args:
        variables = [ request.args.get('variable') ]

    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        title='Measurements for ' + job.input.feature_ids[feature_num] + " by " + ", ".join(variables),
        ylabel='Measurement')
    
    assignments = schema.possible_assignments(variables)

    x = np.arange(len(assignments))
    width = 0.8

    y = []
    grps = [schema.indexes_with_assignments(a) for a in assignments]
    names = [assignment_name(a) for a in assignments]
    y = [ np.mean(measurements[g]) for g in grps]
    err = [ np.std(measurements[g]) for g in grps]
    ax.bar(x, y, yerr=err, color='y')
    plt.xticks(x+width/2., names, rotation=70)

    return figure_response(fig)


@app.route("/features/<feature_num>")
def feature(feature_num):
    job = app.job
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

    adjusted=np.array(pade.conf.adjust_num_diff(mean_perm_count, unperm_count, len(job.input.table)))

    new_scores = (unperm_count - adjusted) / unperm_count

    max_stat = job.results.bins[..., -2]
    print "Max stat", max_stat
    return render_template(
        "feature.html",
        feature_num=feature_num,
        feature_id=job.input.feature_ids[feature_num],
        measurements=job.input.table[feature_num],
        sample_names=job.schema.sample_column_names,
        factors=job.schema.factors,
        factor_values=factor_values,
        layout=job.full_model.layout,
        tuning_params=job.settings.tuning_params,
        stats=stats,
        bins=bins,
        num_bins=len(job.results.bins[0]),
        unperm_count=unperm_count,
        mean_perm_count=mean_perm_count,
        adjusted_perm_count=adjusted,
        max_stat=max_stat,
        scores=job.results.feature_to_score[..., feature_num],
        new_scores=new_scores
        )

@app.route("/details/<conf_level>")
def details(conf_level):
    job = app.job

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
        num_pages=num_pages,
        conf_level=conf_level,
        min_score=score,
        indexes=idxs,
        group_names=job.results.group_means.header,
        coeff_names=job.results.coeff_values.header,
        fold_change_group_names=job.results.fold_change.header,
        stat_name=job.settings.stat_name,
        scores=scores[idxs],
        stats=scores[idxs],
        means=job.results.group_means.table[idxs],
        coeffs=job.results.coeff_values.table[idxs],
        feature_ids=job.input.feature_ids[idxs],
        fold_change=job.results.fold_change.table[idxs],
        page_num=page_num)

@app.route("/stat_dist.html")
def stat_dist_plots_page():
    semilogx = request.args.get('semilogx') == 'True'
    return render_template("stat_dist.html", 
                           job=app.job,
                           semilogx=semilogx)

@app.route("/feature_count_and_score_by_stat.html")
def feature_count_and_score_by_stat():
    semilogx = request.args.get('semilogx') == 'True'
    return render_template("feature_count_and_score_by_stat.html", 
                           job=app.job,
                           semilogx=semilogx)

@app.route("/confidence_dist.html")
def confidence_dist():
    return render_template("confidence_dist.html", 
                           job=app.job)

@app.route("/stat_dist/<tuning_param>.png")
def stat_dist_plot(tuning_param):
    max_stat = np.max(app.job.results.raw_stats)
    tuning_param = int(tuning_param)
    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        title=app.job.settings.stat_name + " distribution over features, $\\alpha = " + str(tuning_param) + "$",
        xlabel=app.job.settings.stat_name + " value",
        ylabel="Features",
        xlim=(0, max_stat))

    plt.hist(app.job.results.raw_stats[tuning_param], log=False, bins=250)
    return figure_response(fig)

@app.route("/bin_to_score.png")
def bin_to_score_plot():
    data = app.job.results.bin_to_score
    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        title="Confidence by stat value value",
        xlabel="Statistic value",
        ylabel="Confidence")

    for i, param in enumerate(app.job.settings.tuning_params):
        ax.plot(app.job.results.bins[i, :-1], data[i], label=str(param))

    if request.args.get('semilogx') == 'True':
        ax.semilogx(base=10)
    ax.legend(loc='lower right')

    return figure_response(fig)

@app.route("/bin_to_features.png")
def bin_to_features_plot():

    params = app.job.settings.tuning_params
    if 'tuning_param_idx' in request.args:
        params = [ params[int(request.args.get('tuning_param_idx'))] ]

    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        title='Features count by statistic value',
        xlabel='Statistic value',
        ylabel='Features')
    job = app.job
    for i, param in enumerate(params):
        ax.plot(job.results.bins[i, :-1], job.bin_to_mean_perm_count[i], '--', label=str(param) + " permuted")
        ax.plot(job.results.bins[i, :-1], job.bin_to_unperm_count[i], label=str(param) + " unpermuted")
    if request.args.get('semilogx') == 'True':
        ax.semilogx(base=10)
    ax.legend(loc='upper right')
    return figure_response(fig)

@app.route("/conf_dist.png")
def conf_dist_plot():
    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        title="Feature count by confidence score",
        xlabel="Confidence score",
        ylabel="Features")
    ax.plot(app.job.summary.bins, app.job.summary.counts)
    return figure_response(fig)
    

@app.route("/score_dist_for_tuning_params.png")
def score_dist_by_tuning_param():
    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        title='Features by confidence score',
        xlabel='Confidence',
        ylabel='Features')

    lines = []
    labels = []

    params = app.job.settings.tuning_params
    if 'tuning_param_idx' in request.args:
        params = [ int(request.args.get('tuning_param_idx')) ]

    for i, alpha in enumerate(params):
        bins = np.arange(0.5, 1.0, 0.01)
        hist = cumulative_hist(app.job.feature_to_score[i], bins)
        lines.append(ax.plot(bins[:-1], hist, label=str(alpha)))
        labels.append(str(alpha))
    ax.legend(loc='upper right')

    return figure_response(fig)

