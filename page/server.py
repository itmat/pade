import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from page.stat import cumulative_hist
import numpy as np
from flask import Flask, render_template, make_response, request
from page.db import DB
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

@app.route("/conf_dist.png")
def conf_dist_png():
    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        title="Feature count by confidence score",
        xlabel="Confidence score",
        ylabel="Features")
    ax.plot(app.db.summary_bins, app.db.summary_counts)
    return figure_response(fig)

@app.route("/stat_dist/<tuning_param>.png")
def plot_stat_dist(tuning_param):
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
        title="Score by bin",
        xlabel="Bin",
        ylabel="Score")

    for i, param in enumerate(app.db.tuning_params):
        ax.plot(data[i], label=str(param))

    ax.legend(loc='lower right')
    ax.semilogx(base=10)
    return figure_response(fig)

@app.route("/bin_to_features.png")
def bin_to_features_plot():

    params = app.db.tuning_params
    if 'tuning_param_idx' in request.args:
        params = [ params[int(request.args.get('tuning_param_idx'))] ]

    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        title='Features by bin',
        xlabel='bin',
        ylabel='features')
    db = app.db
    for i, param in enumerate(params):
        ax.plot(db.bin_to_mean_perm_count[i], label=str(param) + " permuted")
        ax.plot(db.bin_to_unperm_count[i], label=str(param) + " unpermuted")
    ax.semilogx(base=10)
    ax.legend(loc='upper right')
    return figure_response(fig)
    

@app.route("/score_dist_for_tuning_params.png")
def score_dist_by_tuning_param():
    fig = plt.figure()
    ax = fig.add_subplot(
        111)

    lines = []
    labels = []

    params = app.db.tuning_params
    if 'tuning_param_idx' in request.args:
        params = [ int(request.args.get('tuning_param_idx')) ]

    for i, alpha in enumerate(params):
        bins = np.arange(0.5, 1.0, 0.01)
        hist = cumulative_hist(app.db.feature_to_score[i], bins)
        print "Shape of bins is", np.shape(bins)
        print "Shape of hist is", np.shape(hist)
        lines.append(ax.plot(bins[:-1], hist, label=str(alpha)))
        labels.append(str(alpha))

    png_output = StringIO.StringIO()
    ax.legend(loc='upper right')
    fig.savefig(png_output)
    response = make_response(png_output.getvalue())
    response.headers['Content-Type'] = 'image/png'
    return response

