.. PaGE documentation master file, created by
   sphinx-quickstart on Tue Jan 15 14:32:08 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PaGE's documentation!
================================

Contents:

.. toctree::
   :maxdepth: 2

   installing
   algorithm
   2013-01-15_status
   api

Glossary
========

layout
  A list of lists that specifies groups of indexes into the data
  table. For example::

    [[0,1,2,3],
     [4,5,6,7],
     [8,9,10,11]]

  specifies three groups, each with four columns.

factor
  The name of a variable associated with one of the sample
  columns. For example, you might have factors called "treated",
  "sex", "age", or "batch".

factor value
  A value that one of the factors can take on. For example, "sex"
  might take on values "male" and "female".

schema
  Describes the column structure of the input files. Specifies which
  column contains feature ids (such as gene names), and which columns
  contain expression counts or intensities. Defines the factor of
  interest and associates factor value with each sample.
  

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

