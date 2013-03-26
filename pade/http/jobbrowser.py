from __future__ import absolute_import, print_function, division
from werkzeug import secure_filename
import logging
from flask.ext.wtf import (
    Form, StringField, 
    FileField, SubmitField, 
    TextAreaField)
from flask import Blueprint, render_template, request, session, redirect, url_for, flash

runner = Blueprint(
    'job_browser', __name__,
    template_folder='templates')

mdb = None

class JobImportForm(Form):
    job_file = FileField('Job file:')
    name        = StringField('Name')
    description = TextAreaField('Description (optional)')
    submit      = SubmitField("Upload")

@runner.route("/")
def index():
    return render_template("index.html")


@runner.route("/import_job", methods=['GET', 'POST'])
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


