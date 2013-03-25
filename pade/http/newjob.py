from __future__ import absolute_import, print_function, division

from flask import Blueprint, render_template, request, session, redirect, url_for
from flask.ext.wtf import (
    Form, StringField, Required, FieldList, SelectField, 
    FileField, SubmitField, BooleanField, IntegerField, FloatField,
    TextAreaField, FormField)
from itertools import repeat
from collections import defaultdict, OrderedDict
from pade.analysis import assignment_name
from pade.model import Job, Settings, Schema

import pade.model
import logging, os, csv, celery
import numpy as np

bp = Blueprint(
    'new_job', __name__,
    template_folder='templates')

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
        raw_file = mdb.input_file(self.input_file_id)

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
        return mdb.input_file(self.input_file_id)

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

@bp.route("/clear_workflow")
def clear_workflow():
    if 'workflow' in session:
        del session['workflow']
    return redirect(url_for('index'))


@bp.route("/select_input_file")
def select_input_file():
    logging.info("Setting input file id\n")

    input_file_id = int(request.args.get('input_file_id'))

    session['workflow'] = Workflow(input_file_id)
    return redirect(url_for('.column_roles'))


@bp.route("/column_roles", methods=['GET', 'POST'])
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
        return redirect(url_for('.factor_list'))


@bp.route("/factor_list")
def factor_list():

    wf = current_workflow()

    if not wf.can_continue_past_factors:
        return redirect(url_for('.add_factor'))

    return render_template('factors.html',
                           factor_forms=wf.factor_forms,
                           allow_next=True)


@bp.route("/add_factor", methods=['GET', 'POST'])
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
        return redirect(url_for('.factor_list'))


@bp.route("/remove_factor")
def remove_factor():

    factor = request.args.get('factor')
    wf = current_workflow()
    wf.remove_factor(factor)
    session.modified = True

    return redirect(url_for('.factor_list'))


@bp.route("/column_labels", methods=['GET', 'POST'])
def column_labels():

    wf = current_workflow()
    field_names = np.array(wf.field_names)
    roles       = np.array(wf.column_roles)
    sample_field_names = field_names[roles == 'sample']

    if request.method == 'POST':
        form = ColumnLabelsForm(request.form)
        wf.column_label_form = form
        logging.info("I got " + str(len(form.assignments)) + " assignments")
        return redirect(url_for('.setup_job_factors'))

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
    

@bp.route("/setup_job_factors", methods=['GET', 'POST'])
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
        return redirect(url_for('.job_settings'))


@bp.route("/other_settings", methods=['GET', 'POST'])
def job_settings():

    wf = current_workflow()

    if request.method == 'POST':
        wf.other_settings_form = JobSettingsForm(request.form)
        return redirect(url_for('.job_confirmation'))

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

@bp.route("/confirmation")
def job_confirmation():
    wf = current_workflow()

    return render_template(
        'job_confirmation.html',
        input_file_meta=wf.input_file_meta,
        schema=wf.schema,
        settings=wf.settings)


@bp.route("/submit")
def submit_job():
    wf = current_workflow()

    # Create the file for the schema and add it to the db
    schema_meta = mdb.add_schema(
        'Schema', 'Comments', wf.schema, wf.input_file_meta)

    # Add the job to the database
    job_meta = mdb.add_job(
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
    mdb.add_task_id(job_meta, result.task_id)
    clear_workflow()
    return redirect(url_for('job.job_details', job_id=job_meta.obj_id))
