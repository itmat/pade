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

import pade.jobdetails
import pade.newjob

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
pade.jobdetails.mdb = app.mdb
pade.newjob.mdb = app.mdb


app.register_blueprint(pade.jobdetails.bp, url_prefix="/jobs/<job_id>/")
app.register_blueprint(pade.newjob.bp, url_prefix="/new_job/")



def datetime_format(dt):
    """Jinja2 filter for formatting datetime objects."""

    return dt.strftime('%F %R')
    
app.jinja_env.filters['datetime'] = datetime_format

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/schemas")
def schema_list():
    return render_template(
        'schemas.html',
        schema_metas=app.mdb.all_schemas())


@app.route("/raw_files/<raw_file_id>")
def input_file_details(raw_file_id):
    raw_file = app.mdb.input_file(raw_file_id)

    fieldnames = []
    rows = []
    max_rows = 10
    with open(raw_file.path) as infile:
        csvfile = csv.DictReader(infile, delimiter="\t")
        fieldnames = csvfile.fieldnames    
        for i, row in enumerate(csvfile):
            rows.append(row)
            if i == max_rows:
                break
    
    return render_template(
        'input_file.html',
        raw_file=raw_file,
        fieldnames=fieldnames,
        sample_rows=rows)
        

@app.route("/inputfiles")
def input_file_list():
    files = app.mdb.all_input_files()
    files = sorted(files, key=lambda f:f.obj_id, reverse=True)
    return render_template(
        'input_files.html',
        input_file_metas=files)


@app.route("/upload_raw_file", methods=['GET', 'POST'])
def upload_raw_file():

    form = InputFileUploadForm(request.form)
    
    if request.method == 'GET':
        return render_template('upload_raw_file.html', form=form)

    elif request.method == 'POST':

        file = request.files['input_file']
        filename = secure_filename(file.filename)
        logging.info("Adding input file to meta db")
        meta = app.mdb.add_input_file(name=filename, stream=file, description=form.description.data)

        return redirect(url_for('input_file_list'))


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
