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
import celery
import contextlib
import os

from collections import defaultdict, OrderedDict
from itertools import repeat
from celery.result import AsyncResult
from redis import Redis
from bisect import bisect
from flask import (
    Flask, render_template, make_response, request, session, redirect, 
    url_for, flash, send_file)
from flask.ext.wtf import (
    Form, StringField, Required, FieldList, SelectField, 
    FileField, SubmitField, BooleanField, IntegerField, FloatField,
    TextAreaField, FormField)
from werkzeug import secure_filename
from pade.analysis import assignment_name
from pade.stat import cumulative_hist, adjust_num_diff
from pade.model import Job, Settings, Schema
from pade.metadb import MetaDB

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


class Workflow():
    def __init__(self, input_file_id):
        self.input_file_id = input_file_id
        self.column_roles_form = None
        self.factor_forms = []
        self.column_labels_form = None
        self.job_factor_form = None
        self.other_settings_form = None

    @property
    def field_names(self):
        raw_file = app.mdb.input_file(self.input_file_id)

        with open(raw_file.path) as infile:
            csvfile = csv.DictReader(infile, delimiter="\t")
            return csvfile.fieldnames    

    @property
    def can_continue_past_factors(self):
        return len(self.factor_forms) > 0

    def remove_factor(self, factor):
        keep = lambda x: x.factor_name.data != factor
        self.factor_forms = filter(keep, self.factor_forms)

    @property
    def column_roles(self):
        return [ x.data for x in self.column_roles_form.roles ]

    @property
    def factor_values(self):
        factor_to_values = OrderedDict()

        for factor_form in self.factor_forms:
            factor = factor_form.factor_name.data
            values = [x.data for x in factor_form.possible_values if len(x.data) > 0]
            factor_to_values[factor] = values
        return factor_to_values

    @property
    def input_file_meta(self):
        return app.mdb.input_file(self.input_file_id)

    @property
    def schema(self):

        wf = current_workflow()
        columns = np.array(wf.field_names)
        roles   = np.array(wf.column_roles)
        factors = self.factor_values.keys()
        
        logging.info("Initializing schema with columns " + str(columns) + " and roles " + str(roles))

        if columns is None or roles is None or len(columns) == 0 or len(roles) == 0:
            raise Exception("I can't create a schema without columns or roles")

        schema = Schema(map(str, columns),
                        map(str, wf.column_roles))

        for factor, values in self.factor_values.items():
            schema.add_factor(str(factor), map(str, values))

        counter = 0

        for i, c in enumerate(columns[roles == 'sample']):
            for j, f in enumerate(factors):
                try:
                    value = self.column_label_form.assignments[counter].data
                except IndexError as e:
                    raise Exception("No assignment " + str(counter))
                schema.set_factor(str(c), str(f), str(value))
                counter += 1

        return schema

    @property
    def settings(self):
        
        form            = self.other_settings_form
        job_factor_form = self.job_factor_form

        tuning_params=[float(x) for x in form.tuning_params.data.split(" ")]

        condition_vars = []
        block_vars = []
    
        for i, factor in enumerate(self.schema.factors):
            value = job_factor_form.factor_roles[i].data
            if value == 'condition':
                condition_vars.append(factor)
            elif value == 'block':
                block_vars.append(factor)    

        return Settings(
            condition_variables=condition_vars,
            block_variables=block_vars,
            stat_class = str(form.statistic.data),
            equalize_means = form.equalize_means.data,
            num_bins = form.bins.data,
            num_samples = form.permutations.data,
            sample_from_residuals = form.sample_from_residuals.data,
            sample_with_replacement = form.sample_with_replacement.data,
            summary_min_conf = form.summary_min_conf_level.data,
            summary_step_size = form.summary_step_size.data,
            tuning_params = tuning_params)

########################################################################
###
### Form classes
###

class InputFileUploadForm(Form):
    input_file = FileField('Input file')
    description = TextAreaField('Description (optional)')
    submit     = SubmitField("Upload")

class JobImportForm(Form):
    job_file = FileField('Job file:')
    name        = StringField('Name')
    description = TextAreaField('Description (optional)')
    submit      = SubmitField("Upload")

class ColumnRolesForm(Form):
    roles = FieldList(
        SelectField(choices=[('feature_id', 'Feature ID'),
                             ('sample',     'Sample'),
                             ('ignored',    'Ignored')]))
    submit = SubmitField()

    def cols_with_role(self, role):
        has_role = lambda x: x.data == role
        return filter(has_role, self.roles)

class NewFactorForm(Form):
    factor_name = StringField('Factor name', validators=[Required()])
    possible_values = FieldList(StringField(''))
    submit = SubmitField()

class ColumnLabelsForm(Form):
    assignments = FieldList(SelectField())


class JobFactorForm(Form):
    factor_roles = FieldList(
        SelectField(choices=[('block', 'Block'),
                             ('condition',     'Condition'),
                             ('ignored',    'Ignored')]))
    submit = SubmitField()

class JobSettingsForm(Form):

    statistic    = SelectField('Statistic', choices=[('FStat', 'F'), 
                                                     ('OneSampleDifferenceTStat', 'One-Sample T'),
                                                     ('MeansRatio', 'Means Ratio')])
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
        'Use bootstrapping (instead of permutation test)', 
        validators=[Required()],
        default=pade.model.DEFAULT_SAMPLE_WITH_REPLACEMENT)

    equalize_means = BooleanField(
        'Equalize means', 
        validators=[Required()],
        default=pade.model.DEFAULT_EQUALIZE_MEANS)

    summary_min_conf_level = FloatField(
        'Minimum confidence level', 
        validators=[Required()],
        default=pade.model.DEFAULT_SUMMARY_MIN_CONF)

    summary_step_size = FloatField(
        'Summary step size', 
        validators=[Required()],
        default=pade.model.DEFAULT_SUMMARY_STEP_SIZE)
    
    tuning_params = StringField(
        'Tuning parameters', 
        validators=[Required()],
        default=' '.join(map(str, pade.model.DEFAULT_TUNING_PARAMS)))

    submit = SubmitField()


###
### Helpers.
### 


def current_workflow():
    """Return the Workflow stored in the session."""
    return session['workflow']


def datetime_format(dt):
    """Jinja2 filter for formatting datetime objects."""

    return dt.strftime('%F %R')
    
def figure_response(fig):
    """Turns a matplotlib figure into an HTTP response."""
    png_output = StringIO.StringIO()
    fig.savefig(png_output)
    response = make_response(png_output.getvalue())
    response.headers['Content-Type'] = 'image/png'
    return response

def load_job(job_id):
    """Load the Job object with the given meta job id."""
    job_meta = app.mdb.job(job_id)
    return pade.tasks.load_job(job_meta.path)


app.jinja_env.filters['datetime'] = datetime_format


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/jobs/<job_id>")
def job_details(job_id):

    job_meta = app.mdb.job(job_id)

    if job_meta.imported:
        task = None
    else:
        task_ids = app.mdb.get_task_ids(job_meta)
        tasks = [ AsyncResult(x) for x in task_ids ]

        if len(tasks) != 1:
            raise Exception("I got " + str(len(tasks)) +
                            " tasks for job " + 
                            str(job_id) + "; this should never happen")
        task = tasks[0]

    if job_meta.imported or task.status == 'SUCCESS':
        job = load_job(job_id)
        return render_template("job.html", job_id=job.job_id, job=job)

    else:
        return render_template(
            'job_status.html',
            job_id=job_id,
            status=task.status)


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


@app.route("/new_job/clear_workflow")
def clear_workflow():
    if 'workflow' in session:
        del session['workflow']
    return redirect(url_for('index'))


@app.route("/new_job/select_input_file")
def select_input_file():
    logging.info("Setting input file id\n")

    input_file_id = int(request.args.get('input_file_id'))

    session['workflow'] = Workflow(input_file_id)
    return redirect(url_for('column_roles'))


@app.route("/new_job/column_roles", methods=['GET', 'POST'])
def column_roles():

    wf = current_workflow()

    if request.method == 'GET':

        fieldnames = current_workflow().field_names

        form = ColumnRolesForm()
        for i, fieldname in enumerate(fieldnames):
            entry = form.roles.append_entry()
            entry.label = fieldname
            entry.data = 'feature_id' if i == 0 else 'sample'

        return render_template(
            'column_roles.html',
            form=form,
            filename=wf.input_file_meta.name)

    elif request.method == 'POST':
        wf.column_roles_form = ColumnRolesForm(request.form)
        return redirect(url_for('factor_list'))


@app.route("/new_job/factor_list")
def factor_list():

    wf = current_workflow()

    if not wf.can_continue_past_factors:
        return redirect(url_for('add_factor'))

    return render_template('factors.html',
                           factor_forms=wf.factor_forms,
                           allow_next=True)


@app.route("/new_job/add_factor", methods=['GET', 'POST'])
def add_factor():

    wf = current_workflow()

    if request.method == 'GET':
        form = NewFactorForm()

        sample_entries = wf.column_roles_form.cols_with_role('sample')

        for i in range(len(sample_entries) // 2):
            form.possible_values.append_entry()

        return render_template('add_factor.html',
                               form=form,
                               allow_next=wf.can_continue_past_factors)

    elif request.method == 'POST':
        wf.factor_forms.append(NewFactorForm(request.form))
        session.modified = True
        return redirect(url_for('factor_list'))


@app.route("/new_job/remove_factor")
def remove_factor():

    factor = request.args.get('factor')
    wf = current_workflow()
    wf.remove_factor(factor)
    session.modified = True

    return redirect(url_for('factor_list'))


@app.route("/new_job/column_labels", methods=['GET', 'POST'])
def column_labels():

    wf = current_workflow()
    field_names = np.array(wf.field_names)
    roles       = np.array(wf.column_roles)
    sample_field_names = field_names[roles == 'sample']

    if request.method == 'POST':
        form = ColumnLabelsForm(request.form)
        wf.column_label_form = form
        logging.info("I got " + str(len(form.assignments)) + " assignments")
        return redirect(url_for('setup_job_factors'))

    else:
        factor_to_values = wf.factor_values

        form = ColumnLabelsForm()
        
        for col in sample_field_names:
            for factor in factor_to_values:
                entry = form.assignments.append_entry()
                entry.choices = [ (x, x) for x in factor_to_values[factor] ]

        return render_template("column_labels.html",
                               form=form,
                               column_names=sample_field_names,
                               factors=factor_to_values.keys(),
                               input_file_meta=wf.input_file_meta)
    

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

@app.route("/new_job/setup_job_factors", methods=['GET', 'POST'])
def setup_job_factors():

    wf = current_workflow()
    if request.method == 'GET':

        form = JobFactorForm()

        for i, factor in enumerate(wf.factor_values):
            entry = form.factor_roles.append_entry()
            entry.label = factor
            entry.data = 'condition' if i == 0 else 'block'

        return render_template('setup_job_factors.html', form=form)

    elif request.method == 'POST':
        wf.job_factor_form = JobFactorForm(request.form)
        session.modified = True
        return redirect(url_for('job_settings'))


@app.route("/new_job/other_settings", methods=['GET', 'POST'])
def job_settings():

    wf = current_workflow()

    if request.method == 'POST':
        wf.other_settings_form = JobSettingsForm(request.form)
        return redirect(url_for('job_confirmation'))

    else:

        schema = wf.schema

        form = JobSettingsForm()

        form.bins.data                    = pade.model.DEFAULT_NUM_BINS
        form.permutations.data            = pade.model.DEFAULT_NUM_SAMPLES
        form.sample_from_residuals.data   = False
        form.sample_with_replacement.data = False
        form.equalize_means.data = False
        form.summary_min_conf_level.data  = pade.model.DEFAULT_SUMMARY_MIN_CONF
        form.summary_step_size.data = pade.model.DEFAULT_SUMMARY_STEP_SIZE

        form.tuning_params.data = " ".join(map(str, pade.model.DEFAULT_TUNING_PARAMS))
        wf.other_settings_form = form

        return render_template('setup_job.html', form=form)

@app.route("/new_job/confirmation")
def job_confirmation():
    wf = current_workflow()

    return render_template(
        'job_confirmation.html',
        input_file_meta=wf.input_file_meta,
        schema=wf.schema,
        settings=wf.settings)


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


@app.route("/jobs/<job_id>/features/<feature_num>/interaction_plot")
def interaction_plot(job_id, feature_num):
    job = load_job(job_id)
    feature_num = int(feature_num)
    schema = job.schema
    measurements = job.input.table[feature_num]

    x_var = job.settings.condition_variables[0]
    series_var = job.settings.block_variables[0]
    assignments = schema.possible_assignments([series_var, x_var])

    ticks = schema.factor_values[x_var]

    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        ylabel='Measurement')
    
    for series_name in schema.factor_values[series_var]:
        y = []
        yerr = []
        for i, tick in enumerate(ticks):
            a = { series_var : series_name,
                  x_var : tick }
            idxs = schema.indexes_with_assignments(a)
            values = measurements[idxs]
            y.append(np.mean(values))
            yerr.append(np.std(values))

        ax.errorbar(x=np.arange(len(ticks)), y=y, yerr=yerr, label=series_name)

    margin = 0.25

    ax.set_xlim(( 0 - margin, len(ticks) - 1 + margin))
    ax.set_xticks(np.arange(len(ticks)))
    ax.set_xticklabels(ticks)

    ax.legend()
    ax.xlabel = x_var
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
        title='Measurements for feature ' + job.input.feature_ids[feature_num] + " by " + ", ".join(variables),
        ylabel='Measurement')
    
    assignments = schema.possible_assignments(variables)

    x = np.arange(len(assignments))
    width = 0.8

    y = []
    grps = [schema.indexes_with_assignments(a) for a in assignments]
    names = [", ".join(a.values()) for a in assignments]
    y = [ np.mean(measurements[g]) for g in grps]
    err = [ np.std(measurements[g]) for g in grps]
    ax.bar(x, y, yerr=err, color='y')
    plt.xticks(x+width/2., names, rotation=70)

    return figure_response(fig)


@app.route("/jobs/<job_id>/features/<feature_num>")
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
        stat_name=job.settings.stat_class.NAME,
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
        title=job.settings.stat_class + " distribution over features, $\\alpha = " + str(tuning_param) + "$",
        xlabel=job.settings.stat_class.NAME + " value",
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
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos=0: "{:.0f}%".format(x * 100)))
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

@app.route("/new_job/submit")
def submit_job():
    wf = current_workflow()

    # Create the file for the schema and add it to the db
    schema_meta = app.mdb.add_schema(
        'Schema', 'Comments', wf.schema, wf.input_file_meta)

    # Add the job to the database
    job_meta = app.mdb.add_job(
        name='', description='', raw_file_meta=wf.input_file_meta,
        schema_meta=schema_meta)
    
    steps = pade.tasks.steps(
        infile_path=os.path.abspath(wf.input_file_meta.path),
        schema=wf.schema,
        settings=wf.settings,
        sample_indexes_path=None,
        path=os.path.abspath(job_meta.path),
        job_id=job_meta.obj_id)

    chained = celery.chain(steps)
    result = chained.apply_async()
    app.mdb.add_task_id(job_meta, result.task_id)
    clear_workflow()
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


#def schema_to_column_roles_form(schema):
#    form = ColumnRolesForm()
#    for i, col in enumerate(schema.column_names):
#        entry = form.roles.append_entry()
#        entry.label = col
#        entry.data = schema.column_roles[i]
#    
#    return form
