from __future__ import absolute_import, print_function, division
from werkzeug import secure_filename
import logging
from flask.ext.wtf import (
    Form, StringField, 
    FileField, SubmitField, 
    TextAreaField)
from flask import Blueprint, render_template, request, session, redirect, url_for, flash

bp = Blueprint(
    'job_browser', __name__,
    template_folder='templates')

mdb = None

class JobImportForm(Form):
    job_file = FileField('Job file:')
    name        = StringField('Name')
    description = TextAreaField('Description (optional)')
    submit      = SubmitField("Upload")

@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/import_job", methods=['GET', 'POST'])
def import_job():

    form = JobImportForm(request.form)

    if request.method == 'GET':
        return render_template('import_job.html', form=form)

    elif request.method == 'POST':
        file = request.files['job_file']
        filename = secure_filename(file.filename)
        logging.info("Importing job")
        job_meta = mdb.add_job(name=form.name,
                                   description=form.description,
                                   stream=file)
        
        flash("Imported job")
        return redirect(url_for('job.job_details', job_id=job_meta.obj_id))


@bp.route("/jobs/")
def job_list():
    job_metas = mdb.all_jobs()
    job_metas = sorted(job_metas, key=lambda f:f.dt_created, reverse=True)
    return render_template('jobs.html', jobs=job_metas)
