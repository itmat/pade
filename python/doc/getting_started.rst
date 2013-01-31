Getting Started
===============

Prerequisites
-------------

PaGE requires a number of libraries that aren't built in to Python:

* jinja2
* matplotlib
* numpy
* scipy
* yaml

If you are using Linux, you may be able to install these packages
quite easily using pip (http://pypi.python.org/pypi/pip) or another
package management tool.

If you are using a Mac, installing scipy can be tricky. The easiest
way to get up and running may be to use a Python distribution that has
scientific computing libraries pre-packaged with it. You should be
able to use the *Enthought Python Distribution*, available here:
http://www.enthought.com/products/epd.php.

Obtaining
---------

You can obtain PaGE by cloning or forking it from github here:
https://github.com/itmat/PaGE. 

.. NOTE::
   We need to create a tarball and put it somewhere to be downloaded.

Installing
----------

Once you've obtained PaGE, you can either run it from the directory
where you've unpacked it, or install it globally. If you run it from a
local directory, you'll need to update your PYTHONPATH environment
variable to point to the root of the PaGE directory. For example, if
you've downloaded PaGE to ~/projects/PaGE, you might want to add ::

  export PYTHONPATH=$PYTHONPATH:$HOME/projects/PaGE

to your ~/.profile or ~/.bashrc file.

You can install it simply by running ``python setup.py install``.  Note
that you may need to run this as root or under sudo.
   
Running a sample job
--------------------

We recommend running a small job using the sample data provided in the
PaGE distribution in order to familiarize yourself with the program
before attempting to run it on your own data.

The PaGE distribution includes a couple very small sample data files,
in the ``sample_data`` directory.

Input files
^^^^^^^^^^^

The input to any PaGE job (at this time) is a tab-delimited file. The
file must contain a header row, plus one row for each feature. There
should be one column that contains a unique identifier for the
features (for example a gene id). Then each of the samples should have
their expression values for each feature in its own column. So for
example if you 2 conditions, each with 4 replicates, and 1000
features, you would have a tab file with a header row plus 1000 data
rows, with 9 columns (1 for the feature id and 8 for the expression
data).

The names of the columns do not matter, except that each column's name
must be unique. 

Output files
^^^^^^^^^^^^

When PaGE is done it will place all of its output data and HTML
reports in a directory. The default location is ``pageseq_out``, but
this can be changed with the ``--directory`` option.

Creating the schema
^^^^^^^^^^^^^^^^^^^

The first step of any PaGE job is to run ``page setup`` on the input
file, which will create a "schema" file that you will then edit to
describe the grouping of columns in the input file. You run ``page
setup`` and provide the input file on the command line, plus a
``--factor`` argument for each factor that you want to use to group
the columns. For example, suppose say we are trying to find genes that
are differentially expressed due to some treatment, and we want to
treat gender as a "nuisance" variables. So we have two factors:
"treated" and "gender". We would setup the job as follows::

  page setup --factor gender --factor treated sample_data/sample_data_4_class.txt

This will read in the input file and create a skeleton "schema" file
based on it, in pageseq_out/schema.yaml. We then need to edit this
file to list the values available for each of the two factors, and to
assign those factor values to each of the sample column names.

First, in the very top section of the pageseq_out/schema.yaml file,
list the valid values for the factors. Change it to look like this::

  factors:
  - name: treated
    values: [no, yes]
  - name: gender
    values: [male, female]

Next, look for the section called ``sample_factor_mapping``. This
lists each sample column in the input file, like this::

  sample_factor_mapping:
    c0r1:
      gender: null
      treated: null
    c0r2:
      gender: null
      treated: null
  ...

You will need to edit the settings for each column to indicate the
gender and whether or not it was treated::

  sample_factor_mapping:
    c0r1:
      gender: male
      treated: no
    c0r2:
      gender: male
      treated: no
  ...
    c3r4:
      gender: female
      treated: yes

Running the analysis
^^^^^^^^^^^^^^^^^^^^

Once you have created the schema file, you are ready to run the
analysis, using ``page run``. You'll need to specify a couple options,
most importantly ``--full-model`` and optionally ``--reduced-model``.

Full model
""""""""""

``--full-model`` allows you to specify a formula that indicates which
variables should be considered, and whether or not you want to compute
coefficients for interactions between those variables. If you just
have one factor, or if you want to ignore all but one factor, you
would just provide something like ``--full-model treated``. If you
want to consider two variables, say "treated" and "gender", and you
want to look for interaction effects, you would use::

  --full-model "treated * gender"

If you only want to consider main effects (not interactions), you would use:

  --full-model "treated + gender"

Reduced model
"""""""""""""

If you have more than one variable in the full model, you may specify
a reduced model, which must be a subset of the variables in the full
model. The null hypothesis tested by PaGE is that the variables in the
reduced model describe the data as well as the variables in the full model.

.. NOTE::
   That's a terrible description...

For example, if your full model is "treated * gender" and you want to
consider the effects of treatment only, then your reduced model would
simply be "gender". If your full model is "treated" (without
considering gender at all), you would not provide a reduced model.

.. NOTE::
   This could use work.

Default settings
""""""""""""""""

The simplest PaGE job for our 4-class sample input would be something like::

  page run --full-model "treated * gender" --reduced-model gender

This should take less than a minute, and should print a message
telling you the location of the HTML reports.

Interesting options
"""""""""""""""""""

By default, PaGE computes the false discovery rate by using a
permutation test with the f-statistic. You can change the method used
for computing the false discovery rate with the "--sample-method" and
"--sample-from" options. This allows you to do bootstrapping instead
of permutation, and to sample from either the raw data values or from
the residuals of the data values (from the means predicted by the
reduced model). Please see ``page run -h`` for more details.

You can change the number of samples used for bootstrapping (or the
permutation test) with ``--num-samples`` or ``-R``.

By default PaGE prints very little output; just a report at the end
showing the distribution of the confidence levels. You can make it be
more verbose with the ``--verbose`` or ``-v`` option. It will print
even more debugging-level output if you give it ``--debug`` or ``-d``.