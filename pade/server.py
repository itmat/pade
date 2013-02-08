# TODO:
#  Pre-generate histograms of stat distributions

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pade.stat import cumulative_hist
import numpy as np
from flask import Flask, render_template, make_response, request
from pade.db import DB
import argparse
import logging 
import StringIO


class PadeApp(Flask):

    def __init__(self):
        super(PadeApp, self).__init__(__name__)
        self.db = None

app = PadeApp()

def figure_response(fig):
    png_output = StringIO.StringIO()
    fig.savefig(png_output)
    response = make_response(png_output.getvalue())
    response.headers['Content-Type'] = 'image/png'
    return response

@app.route("/")
def index():
    logging.info("Getting index")
    return render_template("index.html", db=app.db)


class ResultTable:

    def __init__(self,
                 group_names=None,
                 param_names=None,
                 means=None,
                 coeffs=None,
                 stats=None,
                 feature_ids=None,
                 scores=None,
                 min_score=None):
        self.means = means
        self.coeffs = coeffs
        self.group_names = group_names
        self.param_names = param_names
        self.stats = stats
        self.feature_ids = feature_ids
        self.scores = scores
        self.min_score = min_score
        

    def filter_by_score(self, min_score):
        idxs = self.scores > min_score
        best = np.argmax(np.sum(idxs, axis=1))
        idxs = idxs[best]
        stats = self.stats[best]
        scores = self.scores[best]
        return ResultTable(
            group_names=self.group_names,
            param_names=self.param_names,
            means=self.means[idxs],
            coeffs=self.coeffs[idxs],
            stats=stats[idxs],
            feature_ids=self.feature_ids[idxs],
            scores=scores[idxs],
            min_score=min_score)

    def __len__(self):
        return len(self.feature_ids)

    def pages(self, rows_per_page=100):
        for start in range(0, len(self), rows_per_page):
            size = min(rows_per_page, len(self) - start)
            end = start + size

            yield ResultTable(
                group_names=self.group_names,
                param_names=self.param_names,
                means=self.means[start : end],
                coeffs=self.coeffs[start : end],
                stats=self.stats[start : end],
                feature_ids=self.feature_ids[start : end],
                scores=self.scores[start : end])

def assignment_name(a):

    if len(a) == 0:
        return "intercept"
    
    parts = ["{0}={1}".format(k, v) for k, v in a.items()]

    return ", ".join(parts)


@app.route("/details/<conf_level>")
def details(conf_level):
    conf_level = int(conf_level)
    db = app.db

    fitted = db.full_model.fit(db.table)

    results = ResultTable(
        means=db.group_means,
        coeffs=db.coeff_values,
        group_names=db.group_names,
        param_names=db.coeff_names,
        feature_ids=np.array(db.feature_ids),
        stats=db.raw_stats,
        scores=db.feature_to_score)

    score=db.summary_bins[conf_level]
    filtered = results.filter_by_score(score)
    pages = list(filtered.pages(100))
    page_num = 0
    return render_template("conf_level.html",
                    conf_level=conf_level,
                    min_score=score,
                    job=db,
                    results=pages[page_num],
                    page_num=page_num,
                    num_pages=len(pages))

@app.route("/stat_dist.html")
def stat_dist_plots_page():
    semilogx = request.args.get('semilogx') == 'True'
    return render_template("stat_dist.html", 
                           db=app.db,
                           semilogx=semilogx)

@app.route("/feature_count_and_score_by_stat.html")
def feature_count_and_score_by_stat():
    semilogx = request.args.get('semilogx') == 'True'
    return render_template("feature_count_and_score_by_stat.html", 
                           db=app.db,
                           semilogx=semilogx)

@app.route("/confidence_dist.html")
def confidence_dist():
    return render_template("confidence_dist.html", 
                           db=app.db)

@app.route("/stat_dist/<tuning_param>.png")
def stat_dist_plot(tuning_param):
    max_stat = np.max(app.db.raw_stats)
    tuning_param = int(tuning_param)
    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        title=app.db.stat_name + " distribution over features, $\\alpha = " + str(tuning_param) + "$",
        xlabel=app.db.stat_name + " value",
        ylabel="Features",
        xlim=(0, max_stat))

    plt.hist(app.db.raw_stats[tuning_param], log=False, bins=250)
    return figure_response(fig)

@app.route("/bin_to_score.png")
def bin_to_score_plot():
    data = app.db.bin_to_score
    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        title="Confidence by stat value value",
        xlabel="Statistic value",
        ylabel="Confidence")

    for i, param in enumerate(app.db.tuning_params):
        ax.plot(app.db.bins[i, :-1], data[i], label=str(param))

    if request.args.get('semilogx') == 'True':
        ax.semilogx(base=10)
    ax.legend(loc='lower right')

    return figure_response(fig)

@app.route("/bin_to_features.png")
def bin_to_features_plot():

    params = app.db.tuning_params
    if 'tuning_param_idx' in request.args:
        params = [ params[int(request.args.get('tuning_param_idx'))] ]

    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        title='Features count by statistic value',
        xlabel='Statistic value',
        ylabel='Features')
    db = app.db
    for i, param in enumerate(params):
        ax.plot(db.bins[i, :-1], db.bin_to_mean_perm_count[i], '--', label=str(param) + " permuted")
        ax.plot(db.bins[i, :-1], db.bin_to_unperm_count[i], label=str(param) + " unpermuted")
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
    ax.plot(app.db.summary_bins, app.db.summary_counts)
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

    params = app.db.tuning_params
    if 'tuning_param_idx' in request.args:
        params = [ int(request.args.get('tuning_param_idx')) ]

    for i, alpha in enumerate(params):
        bins = np.arange(0.5, 1.0, 0.01)
        hist = cumulative_hist(app.db.feature_to_score[i], bins)
        lines.append(ax.plot(bins[:-1], hist, label=str(alpha)))
        labels.append(str(alpha))
    ax.legend(loc='upper right')

    return figure_response(fig)

