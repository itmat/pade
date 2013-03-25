# TODO:
#  Pre-generate histograms of stat distributions
from __future__ import absolute_import, print_function, division
import matplotlib
matplotlib.use("Agg")
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

import numpy as np
import logging 
import StringIO
import uuid
import shutil
import csv
import pade.redis_session
import padeconfig

import contextlib
import os

from redis import Redis

from flask import (
    Flask, render_template, make_response, request, session, redirect, 
    url_for, flash, send_file)

from flask.ext.wtf import (
    Form, StringField, Required, FieldList, SelectField, 
    FileField, SubmitField, BooleanField, IntegerField, FloatField,
    TextAreaField, FormField)
from werkzeug import secure_filename
from pade.stat import cumulative_hist, adjust_num_diff
from pade.metadb import MetaDB

import pade.http.jobdetails
import pade.http.newjob
import pade.http.inputfile

ALLOWED_EXTENSIONS = set(['txt', 'tab'])

class PadeApp(Flask):

    def __init__(self):
        super(PadeApp, self).__init__(__name__)
        self.job = None
        self.mdb = None
        self.secret_key = ""

app = PadeApp()
app.session_interface = pade.redis_session.RedisSessionInterface(
    Redis(db=padeconfig.DB_SESSION))
app.mdb = MetaDB(padeconfig.METADB_DIR, Redis(db=padeconfig.DB_METADB))

pade.http.jobdetails.mdb = app.mdb
pade.http.newjob.mdb = app.mdb
pade.http.inputfile.mdb = app.mdb

app.register_blueprint(pade.http.jobdetails.bp, url_prefix="/jobs/<job_id>/")
app.register_blueprint(pade.http.newjob.bp,     url_prefix="/new_job/")
app.register_blueprint(pade.http.inputfile.bp,  url_prefix="/input_files/")



def datetime_format(dt):
    """Jinja2 filter for formatting datetime objects."""

    return dt.strftime('%F %R')
    
app.jinja_env.filters['datetime'] = datetime_format

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/import_job", methods=['GET', 'POST'])
def import_job():

    form = JobImportForm(request.form)

    if request.method == 'GET':
        return render_template('import_job.html', form=form)

    elif request.method == 'POST':
        file = request.files['job_file']
        filename = secure_filename(file.filename)
        logging.info("Importing job")
        job_meta = app.mdb.add_job(name=form.name,
                                   description=form.description,
                                   stream=file)
        
        flash("Imported job")
        return redirect(url_for('job_details', job_id=job_meta.obj_id))


@app.route("/jobs")
def job_list():
    job_metas = app.mdb.all_jobs()
    job_metas = sorted(job_metas, key=lambda f:f.dt_created, reverse=True)
    return render_template('jobs.html', jobs=job_metas)

@app.route("/jobs/<job_id>/result_db")
def result_db(job_id):
    job_meta = app.mdb.job(job_id)
    logging.info("Job meta is " + str(job_meta))
    return send_file(job_meta.path, as_attachment=True)
