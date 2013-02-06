import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from page.stat import cumulative_hist
import numpy as np
from flask import Flask, render_template, make_response
from page.db import DB
import argparse
import logging 
import StringIO

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    'db')
args = parser.parse_args()

db = DB(path=args.db)
db.load()

@app.route("/")
def index():
    logging.info("Getting index")
    return render_template("index.html", db=db)

@app.route("/conf_dist.png")
def conf_dist_png():
    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        title="Feature count by confidence score",
        xlabel="Confidence score",
        ylabel="Features")
    ax.plot(db.summary_bins, db.summary_counts)
    png_output = StringIO.StringIO()
    fig.savefig(png_output)
    response = make_response(png_output.getvalue())
    response.headers['Content-Type'] = 'image/png'

    return response

@app.route("/stat_dist/<tuning_param>.png")
def plot_stat_dist(tuning_param):
    max_stat = np.max(db.raw_stats)
    tuning_param = int(tuning_param)
    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        title=db.stat_name + " distribution over features, $\\alpha = " + str(tuning_param) + "$",
        xlabel=db.stat_name + " value",
        ylabel="Features",
        xlim=(0, max_stat))

    plt.hist(db.raw_stats[tuning_param], log=False, bins=250)
    png_output = StringIO.StringIO()
    fig.savefig(png_output)
    response = make_response(png_output.getvalue())
    response.headers['Content-Type'] = 'image/png'

    return response

@app.route("/score_dist_for_tuning_params.png")
def score_dist_by_tuning_param():
    fig = plt.figure()
    ax = fig.add_subplot(
        111)

    lines = []
    labels = []
    for i, alpha in enumerate(db.tuning_params):
        bins = np.arange(0.5, 1.0, 0.01)
        hist = cumulative_hist(db.feature_to_score[i], bins)
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

if __name__ == "__main__":
    app.debug = True
    app.run()

