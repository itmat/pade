# TODO:
#  Pre-generate histograms of stat distributions
from __future__ import absolute_import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import logging 
import StringIO
import uuid
import shutil
import csv
import pade.redis_session
import redisconfig
import celery
import contextlib
import os

from celery.result import AsyncResult
from redis import Redis
from bisect import bisect
from flask import (
    Flask, render_template, make_response, request, session, redirect, 
    url_for, flash)
from flask.ext.wtf import (
    Form, StringField, Required, FieldList, SelectField, 
    FileField, SubmitField, BooleanField, IntegerField, FloatField)
from werkzeug import secure_filename
from pade.analysis import assignment_name
from pade.stat import cumulative_hist, adjust_num_diff
from pade.model import Job, Settings, Schema
from pade.metadb import MetaDB

ALLOWED_EXTENSIONS = set(['txt', 'tab'])
UPLOAD_FOLDER = 'uploads'

class PadeApp(Flask):

    def __init__(self):
        super(PadeApp, self).__init__(__name__)
        self.job = None
        self.mdb = None
        self.secret_key = 'asdf'


app = PadeApp()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.session_interface = pade.redis_session.RedisSessionInterface(
    Redis(db=redisconfig.DB_SESSION))
app.mdb = MetaDB(UPLOAD_FOLDER, Redis(db=redisconfig.DB_METADB))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def figure_response(fig):
    png_output = StringIO.StringIO()
    fig.savefig(png_output)
    response = make_response(png_output.getvalue())
    response.headers['Content-Type'] = 'image/png'
    return response

@app.route("/")
def index():
    return render_template("index.html")

def load_job(job_id):
    job_meta = app.mdb.job_db(job_id)
    return pade.tasks.load_job(job_meta.path)

@app.route("/jobs/<job_id>")
def job_details(job_id):
    job = load_job(job_id)
    print "Job id is ", job.job_id
    return render_template("job.html", job_id=job.job_id, job=job)

@app.route("/schemas")
def schema_list():
    return render_template(
        'schemas.html',
        schema_metas=app.mdb.all_schemas())

@app.route("/inputfiles")
def input_file_list():
    return render_template(
        'input_files.html',
        form=InputFileUploadForm(),
        input_file_metas=app.mdb.all_input_files())

########################################################################
###
### Accessing the session
###

def ensure_job_scratch():
    if 'job_scratch' not in session:
        logging.info("Setting up job scratch")
        session['job_scratch'] = { 
            'schema' : Schema(),
            'settings' : Settings()
            }

def set_job_scratch_schema(schema):
    if 'job_scratch' not in session:
        session['job_scratch'] = {}
    session['job_scratch']['schema'] = schema

def set_job_scratch_input_file_id(input_file_id):
    if 'job_scratch' not in session:
        session['job_scratch'] = {}
    session['job_scratch']['input_file_id'] = input_file_id

def current_scratch_filename():
    scratch = job_scratch()
    if 'filename' in scratch:
        return scratch['filename']
    return None

def current_scratch_schema():
    scratch = job_scratch()
    if 'schema' in scratch:
        return scratch['schema']
    return None

def current_scratch_input_file_meta():
    scratch = job_scratch()
    if 'input_file_id' in scratch:
        return app.mdb.input_file(scratch['input_file_id'])
    return None


@app.route("/clear_job_scratch")
def clear_job_scratch():
    if 'job_scratch' in session:
        del session['job_scratch']
    return redirect(url_for('index'))

@app.route("/select_input_file")
def select_input_file():
    logging.info("Setting input file id\n")

    input_file_id = int(request.args.get('input_file_id'))
    input_file_meta = app.mdb.input_file(input_file_id)

    with open(input_file_meta.path) as infile:
        csvfile = csv.DictReader(infile, delimiter="\t")
        fieldnames = csvfile.fieldnames    

        roles = [ 'sample' for x in fieldnames ]
        if len(roles) > 0:
            roles[0] = 'feature_id'

        set_job_scratch_schema(Schema(fieldnames, roles))
        set_job_scratch_input_file_id(input_file_id)
    return redirect(url_for('column_roles'))


def schema_to_column_roles_form(schema):
    form = ColumnRolesForm()
    for i, col in enumerate(schema.column_names):
        entry = form.roles.append_entry()
        entry.label = col
        entry.data = schema.column_roles[i]
    
    return form

def update_schema_with_column_roles(schema, form):
    names = schema.column_names
    roles = [ e.data for e in form.roles ]
    schema.set_columns(names, roles)

@app.route("/column_roles", methods=['GET', 'POST'])
def column_roles():
    schema = current_scratch_schema()

    if request.method == 'GET':
        return render_template(
            'column_roles.html',
            form=schema_to_column_roles_form(schema),
            filename=current_scratch_input_file_meta().name)

    elif request.method == 'POST':
        update_schema_with_column_roles(schema, ColumnRolesForm(request.form))
        return redirect(url_for('add_factor'))
    

########################################################################
###
### Form classes
###

class NewFactorForm(Form):
    factor_name = StringField('Factor name', validators=[Required()])
    possible_values = FieldList(StringField(''))
    submit = SubmitField()

class ColumnRolesForm(Form):
    roles = FieldList(
        SelectField(choices=[('feature_id', 'Feature ID'),
                             ('sample',     'Sample'),
                             ('ignored',    'Ignored')]))
    submit = SubmitField()

class InputFileUploadForm(Form):
    input_file = FileField('Input file')
    submit     = SubmitField()

class JobFactorForm(Form):
    factor_roles = FieldList(
        SelectField(choices=[('block', 'Block'),
                             ('condition',     'Condition'),
                             ('ignored',    'Ignored')]))
    submit = SubmitField()

class JobSettingsForm(Form):

    statistic    = SelectField('Statistic', choices=[('f_test', 'F-test'), 
                                                     ('one_sample_t_test', 'One-sample t-test'),
                                                     ('means_ratio', 'Ratio of means')])
    bins = IntegerField(
        'Number of bins', 
        validators=[Required()],
        default=pade.model.DEFAULT_NUM_BINS)

    permutations = IntegerField(
        'Maximum number of permutations', 
        validators=[Required()],
        default=pade.model.DEFAULT_NUM_SAMPLES)

    sample_from_residuals = BooleanField(
        'Sample from residuals', 
        validators=[Required()],
        default=pade.model.DEFAULT_SAMPLE_FROM_RESIDUALS)

    sample_with_replacement = BooleanField(
        'Sample with replacement', 
        validators=[Required()],
        default=pade.model.DEFAULT_SAMPLE_WITH_REPLACEMENT)

    equalize_means = BooleanField(
        'Equalize means', 
        validators=[Required()],
        default=pade.model.DEFAULT_EQUALIZE_MEANS)

    min_conf_level = FloatField(
        'Minimum confidence level', 
        validators=[Required()],
        default=pade.model.DEFAULT_MIN_CONF)

    conf_interval = FloatField(
        'Confidence interval', 
        validators=[Required()],
        default=pade.model.DEFAULT_CONF_INTERVAL)
    
    tuning_params = StringField(
        'Tuning parameters', 
        validators=[Required()],
        default=' '.join(map(str, pade.model.DEFAULT_TUNING_PARAMS)))

    submit = SubmitField()
    
def update_schema_with_new_factor(schema, form):
    factor = str(form.factor_name.data)
    values = []
    for val_field in form.possible_values:
        value = str(val_field.data)
        if len(value) > 0:
            values.append(value)
    schema.add_factor(factor, values)    

@app.route("/add_factor", methods=['GET', 'POST'])
def add_factor():

    schema = current_scratch_schema()

    if request.method == 'GET':
        form = NewFactorForm()
        for i in range(10):
            form.possible_values.append_entry()
        factors = schema.factors
        allow_next = len(factors) > 0
        return render_template('add_factor.html',
                               form=form,
                               factors=factors,
                               factor_values=schema.factor_values,
                               allow_next=allow_next)

    elif request.method == 'POST':
        schema = current_scratch_schema()
        form = NewFactorForm(request.form)
        update_schema_with_new_factor(schema, form)
        session.modified = True
        return redirect(url_for('add_factor'))


@app.route("/column_labels", methods=['GET', 'POST'])
def column_labels():

    schema = current_scratch_schema()

    if request.method == 'POST':
        
        for i, col_name in enumerate(schema.column_names):
            for j, factor in enumerate(schema.factors):
                key = 'factor_value_{0}_{1}'.format(i, j)
                if key in request.form:
                    value = str(request.form[key])
                    if len(value) > 0:
                        schema.set_factor(col_name, factor, value)
        
        schema.modified = True
        return redirect(url_for('setup_job_factors'))


    schema = current_scratch_schema()

    factor = request.args.get('factor')

    return render_template("column_labels.html",
                           schema=current_scratch_schema(),
                           input_file_meta=current_scratch_input_file_meta(),
                           factor=factor)
    
def job_scratch():
    if 'job_scratch' not in session:
        return None
    return session['job_scratch']

def current_job_id():
    scratch = job_scratch()
    if scratch is None or 'job_id' not in scratch:
        return None
    return scratch['job_id']



@app.route("/upload_input_file", methods=['POST'])
def upload_input_file():
    
    ensure_job_scratch()
    
    file = request.files['input_file']
    filename = secure_filename(file.filename)
    session['job_scratch']['filename'] = filename

    logging.info("Adding input file to meta db")
    meta = app.mdb.add_input_file(filename, "", file)

    return redirect(url_for('input_file_list'))

@app.route("/setup_job_factors", methods=['GET', 'POST'])
def setup_job_factors():
    schema = current_scratch_schema()

    if request.method == 'GET':

        form = JobFactorForm()
        for factor in schema.factors:
            entry = form.factor_roles.append_entry()
            entry.label = factor
        return render_template(
            'setup_job_factors.html',
            form=form,
            schema=current_scratch_schema())

    elif request.method == 'POST':
        form = JobFactorForm(request.form)
        condition_vars = []
        block_vars = []
        for i, factor in enumerate(schema.factors):
            value = form.factor_roles[i].data
            if value == 'condition':
                condition_vars.append(factor)
            elif value == 'block':
                block_vars.append(factor)

        if 'settings' not in job_scratch():
            job_scratch()['settings'] = Settings()
        job_scratch()['settings'].condition_variables = condition_vars
        job_scratch()['settings'].block_variables = block_vars

        return redirect(url_for('job_settings'))


# Workflow:
# 1. Choose input file
# 2. Pick sample columns
# 3. Add factors (possibly multiple times)
# 4. 

def settings_to_form(settings):
    form = JobSettingsForm()
    
    form.bins.data = settings.num_bins
    form.permutations.data = settings.num_samples
    form.sample_from_residuals.data = settings.sample_from_residuals
    form.sample_with_replacement.data = settings.sample_with_replacement
    form.min_conf_level.data = settings.min_conf
    form.conf_interval.data = settings.conf_interval
    print "Tuning params are", settings.tuning_params
    form.tuning_params.data = " ".join(map(str, settings.tuning_params))
    
    return form

@app.route("/job_settings", methods=['GET', 'POST'])
def job_settings():

    if request.method == 'POST':
        form = JobSettingsForm(request.form)
        tuning_params=[float(x) for x in form.tuning_params.data.split(" ")]
        settings = session['job_scratch']['settings']

        settings.stat_name = str(form.statistic.data)

        settings.num_bins = form.bins.data
        settings.num_samples = form.permutations.data
        settings.sample_from_residuals = form.sample_from_residuals.data
        settings.sample_with_replacement = form.sample_with_replacement.data
        settings.min_conf = form.min_conf_level.data
        settings.conf_interval = form.conf_interval.data
        settings.tuning_params = tuning_params

        return redirect(url_for('job_confirmation'))

    else:
        form = settings_to_form(job_scratch()['settings'])
        return render_template(
            'setup_job.html',
            form=form)

@app.route("/job_confirmation")
def job_confirmation():
    schema   = job_scratch()['schema']
    settings = job_scratch()['settings']
        
    return render_template(
        'job_confirmation.html',
        input_file_meta=current_scratch_input_file_meta(),
        schema=schema,
        settings=settings)

@app.route("/jobs/<job_id>/features/<feature_num>/measurement_scatter")
def measurement_scatter(job_id, feature_num):
    job = load_job(job_id)    
    feature_num = int(feature_num)


    schema = job.schema
    measurements = job.input.table[feature_num]
    
    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        title='Measurements',
        xlabel='Group',
        ylabel='Measurement')

    assignments = schema.possible_assignments(job.full_variables)
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

@app.route("/jobs/<job_id>/mean_vs_std")
def mean_vs_std(job_id):
    job = load_job(job_id)
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

@app.route("/jobs/<job_id>/features/<feature_num>/measurement_bars")
def measurement_bars(job_id, feature_num):
    job = load_job(job_id)    
    feature_num = int(feature_num)
    schema = job.schema
    measurements = job.input.table[feature_num]

    variables = job.full_variables
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


@app.route("/job/<job_id>/features/<feature_num>")
def feature(job_id, feature_num):
    job = load_job(job_id)
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

    adjusted=np.array(adjust_num_diff(mean_perm_count, unperm_count, len(job.input.table)))

    new_scores = (unperm_count - adjusted) / unperm_count

    max_stat = job.results.bins[..., -2]

    return render_template(
        "feature.html",
        feature_num=feature_num,
        feature_id=job.input.feature_ids[feature_num],
        measurements=job.input.table[feature_num],
        sample_names=job.schema.sample_column_names,
        factors=job.schema.factors,
        factor_values=factor_values,
        layout=job.full_layout,
        tuning_params=job.settings.tuning_params,
        stats=stats,
        bins=bins,
        job=job,
        job_id=job.job_id,
        num_bins=len(job.results.bins[0]),
        unperm_count=unperm_count,
        mean_perm_count=mean_perm_count,
        adjusted_perm_count=adjusted,
        max_stat=max_stat,
        scores=job.results.feature_to_score[..., feature_num],
        new_scores=new_scores
        )

@app.route("/jobs/<job_id>/conf_level/<conf_level>")
def details(job_id, conf_level):
    job = load_job(job_id)

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

    print "Feature ids are ", job.input.feature_ids[idxs]

    return render_template(
        "conf_level.html",
        job=job,
        job_id=job.job_id,
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

@app.route("/jobs/<job_id>/stat_dist")
def stat_dist_plots_page(job_id):

    semilogx = request.args.get('semilogx') == 'True'
    return render_template("stat_dist.html", 
                           job_id=job_id,
                           job=load_job(job_id),
                           semilogx=semilogx)

@app.route("/jobs/<job_id>/feature_count_and_score_by_stat.html")
def feature_count_and_score_by_stat(job_id):
    semilogx = request.args.get('semilogx') == 'True'
    return render_template("feature_count_and_score_by_stat.html", 
                           job_id=job_id,
                           job=app.job,
                           semilogx=semilogx)

@app.route("/jobs/<job_id>/confidence_dist")
def confidence_dist(job_id):
    return render_template("confidence_dist.html", 
                           job_id=job_id)

@app.route("/jobs/<job_id>/stat_dist/<tuning_param>.png")
def stat_dist_plot(job_id, tuning_param):
    job = load_job(job_id)
    max_stat = np.max(job.results.raw_stats)
    tuning_param = int(tuning_param)
    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        title=job.settings.stat_name + " distribution over features, $\\alpha = " + str(tuning_param) + "$",
        xlabel=job.settings.stat_name + " value",
        ylabel="Features",
        xlim=(0, max_stat))

    plt.hist(job.results.raw_stats[tuning_param], log=False, bins=250)
    return figure_response(fig)

@app.route("/jobs/<job_id>/bin_to_score.png")
def bin_to_score_plot(job_id):
    job = load_job(job_id)
    data = job.results.bin_to_score
    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        title="Confidence by stat value value",
        xlabel="Statistic value",
        ylabel="Confidence")

    for i, param in enumerate(job.settings.tuning_params):
        ax.plot(job.results.bins[i, :-1], data[i], label=str(param))

    if request.args.get('semilogx') == 'True':
        ax.semilogx(base=10)
    ax.legend(loc='lower right')

    return figure_response(fig)

@app.route("/jobs/<job_id>/bin_to_features.png")
def bin_to_features_plot(job_id):
    job = load_job(job_id)
    params = job.settings.tuning_params
    if 'tuning_param_idx' in request.args:
        params = [ params[int(request.args.get('tuning_param_idx'))] ]

    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        title='Features count by statistic value',
        xlabel='Statistic value',
        ylabel='Features')
    job = job
    for i, param in enumerate(params):
        ax.plot(job.results.bins[i, :-1], job.results.bin_to_mean_perm_count[i], '--', label=str(param) + " permuted")
        ax.plot(job.results.bins[i, :-1], job.results.bin_to_unperm_count[i], label=str(param) + " unpermuted")
    if request.args.get('semilogx') == 'True':
        ax.semilogx(base=10)
    ax.legend(loc='upper right')
    return figure_response(fig)

@app.route("/jobs/<job_id>/conf_dist")
def conf_dist_plot(job_id):
    job = load_job(job_id)
    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        title="Feature count by confidence score",
        xlabel="Confidence score",
        ylabel="Features")
    ax.plot(job.summary.bins, job.summary.counts)
    return figure_response(fig)
    

@app.route("/jobs/<job_id>/score_dist_for_tuning_params.png")
def score_dist_by_tuning_param(job_id):
    job = load_job(job_id)
    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        title='Features by confidence score',
        xlabel='Confidence',
        ylabel='Features')

    lines = []
    labels = []

    params = job.settings.tuning_params
    if 'tuning_param_idx' in request.args:
        params = [ int(request.args.get('tuning_param_idx')) ]

    for i, alpha in enumerate(params):
        bins = np.arange(0.5, 1.0, 0.01)
        hist = cumulative_hist(job.results.feature_to_score[i], bins)
        lines.append(ax.plot(bins[:-1], hist, label=str(alpha)))
        labels.append(str(alpha))
    ax.legend(loc='upper right')

    return figure_response(fig)

@app.route("/submit_job")
def submit_job():
    settings = job_scratch()['settings']
    schema   = current_scratch_schema()
    schema_meta = app.mdb.add_schema('Schema', 'Comments', schema)
    infile_meta = current_scratch_input_file_meta()
    
    job = Job(settings=settings,
              schema=schema,
              results=pade.model.Results())

    job_meta = app.mdb.add_job_db('Job db', 'comments')
    
    steps = pade.tasks.steps(
        infile_path=os.path.abspath(infile_meta.path),
        schema=schema,
        settings=settings,
        sample_indexes_path=None,
        path=os.path.abspath(job_meta.path),
        job_id=job_meta.obj_id)

    chained = celery.chain(steps)
    result = chained.apply_async((job,))
    app.mdb.add_task_id(job_meta, result.task_id)

    return redirect(url_for('job_status', job_id=job_meta.obj_id))

@app.route("/job/<job_id>")
def job_status(job_id):

    job_meta = app.mdb.job_db(job_id)

    task_ids = app.mdb.get_task_ids(job_meta)
    print "Got task ids ", task_ids

    tasks = [ AsyncResult(x) for x in task_ids ]

    if len(tasks) != 1:
        raise Exception("I got " + str(len(tasks)) +
                        " tasks for job " + 
                        str(job_id) + "; this should never happen")
    task = tasks[0]

    return render_template(
        'job_status.html',
        job_id=job_id,
        status=task.status)

@app.route("/jobs")
def job_list():
    job_metas = app.mdb.all_job_dbs()
    print "Job metas are", job_metas
    return render_template('jobs.html', jobs=job_metas)

@contextlib.contextmanager
def figure(path):
    """Context manager for saving a figure.

    Clears the current figure, yeilds, then saves the figure to the
    given path and clears the figure again.

    """
    try:
        logging.debug("Creating figure " + path)
        plt.clf()
        yield
        plt.savefig(path)
    finally:
        plt.clf()

