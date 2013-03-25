from __future__ import absolute_import, print_function, division

from flask.ext.wtf import (
    Form, FileField, TextAreaField, SubmitField)
from flask import Blueprint, render_template, request, make_response, redirect, url_for
import csv, logging
from werkzeug import secure_filename

bp = Blueprint(
    'inputfile', __name__,
    template_folder='templates')

mdb = None

class InputFileUploadForm(Form):
    input_file = FileField('Input file')
    description = TextAreaField('Description (optional)')
    submit     = SubmitField("Upload")


@bp.route("/raw_files/<raw_file_id>")
def input_file_details(raw_file_id):
    raw_file = mdb.input_file(raw_file_id)

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
        

@bp.route("/inputfiles")
def input_file_list():
    files = mdb.all_input_files()
    files = sorted(files, key=lambda f:f.obj_id, reverse=True)
    return render_template(
        'input_files.html',
        input_file_metas=files)


@bp.route("/upload_raw_file", methods=['GET', 'POST'])
def upload_raw_file():

    form = InputFileUploadForm(request.form)
    
    if request.method == 'GET':
        return render_template('upload_raw_file.html', form=form)

    elif request.method == 'POST':

        file = request.files['input_file']
        filename = secure_filename(file.filename)
        logging.info("Adding input file to meta db")
        meta = mdb.add_input_file(name=filename, stream=file, description=form.description.data)

        return redirect(url_for('.input_file_list'))

