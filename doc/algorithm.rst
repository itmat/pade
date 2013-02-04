Algorithm
=========

Confidence estimation
---------------------

There are three methods available for sampling the data in order to
estimate the background distribution of the statistic.

Permutation test
^^^^^^^^^^^^^^^^

With the *permutation test* method, we simply create R different
*legal*, *distinct* permutations of the columns of the raw data, and
then apply the f-test to those permutations.

A permutation is *legal* if for every column, the values of the
nuisance variables (defined by the reduced model) remain the same. Two
orderings are *distinct* if there exists some column such that the
value of one of the variables defined in the full model is different
in the two orderings.

For example, suppose we have the following data, where the full model
consists of "sex" and "treatment", and the reduced model is "sex"
only. The raw ordering of the columns is:

+---------+---------------+---------------+
| Sex     | Male          | Female        |
+---------+-------+-------+-------+-------+
| Treated | Yes   | No    | Yes   | No    |
+=========+===+===+===+===+===+===+===+===+
|         | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
+---------+---+---+---+---+---+---+---+---+

This indicates that columns 0 and 1 are untreated male, 2 and 3 are
treated male, 4 and 5 are untreated female, and 6 and 7 are treated
female.

The arrangement [4, 1, 2, 3, 0, 5, 6, 7] is not legal, because samples
0 and 4 had their "sex" label changed. The arrangements [0, 1, 2, 3,
4, 5, 6, 7] and [1, 0, 2, 3, 4, 5, 6, 7] are not distinct because even
though samples 0 and 1 were swapped, their "treated" label was not
changed, so it results in the same grouping of samples.

In this case, we have 36 legal, distinct permutations of the samples
that conform to the full and reduced layouts:

+--------+---------------+---------------+
| Sex    | Male          | Female        |
+--------+-------+-------+-------+-------+
|Treated | No    | Yes   | No    | Yes   |
+========+===+===+===+===+===+===+===+===+
|        | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
+--------+---+---+---+---+---+---+---+---+
|        | 0 | 1 | 2 | 3 | 4 | 6 | 5 | 7 |
+--------+---+---+---+---+---+---+---+---+
|        | 0 | 1 | 2 | 3 | 4 | 7 | 5 | 6 |
+--------+---+---+---+---+---+---+---+---+
|        | 0 | 1 | 2 | 3 | 5 | 6 | 4 | 7 |
+--------+---+---+---+---+---+---+---+---+
|        | 0 | 1 | 2 | 3 | 5 | 7 | 4 | 6 |
+--------+---+---+---+---+---+---+---+---+
|        | 0 | 1 | 2 | 3 | 6 | 7 | 4 | 5 |
+--------+---+---+---+---+---+---+---+---+
|        | 0 | 2 | 1 | 3 | 4 | 5 | 6 | 7 |
+--------+---+---+---+---+---+---+---+---+
|        | 0 | 2 | 1 | 3 | 4 | 6 | 5 | 7 |
+--------+---+---+---+---+---+---+---+---+
|        | 0 | 2 | 1 | 3 | 4 | 7 | 5 | 6 |
+--------+---+---+---+---+---+---+---+---+
|        | 0 | 2 | 1 | 3 | 5 | 6 | 4 | 7 |
+--------+---+---+---+---+---+---+---+---+
|        | 0 | 2 | 1 | 3 | 5 | 7 | 4 | 6 |
+--------+---+---+---+---+---+---+---+---+
|        | 0 | 2 | 1 | 3 | 6 | 7 | 4 | 5 |
+--------+---+---+---+---+---+---+---+---+
|        | 0 | 3 | 1 | 2 | 4 | 5 | 6 | 7 |
+--------+---+---+---+---+---+---+---+---+
|        | 0 | 3 | 1 | 2 | 4 | 6 | 5 | 7 |
+--------+---+---+---+---+---+---+---+---+
|        | 0 | 3 | 1 | 2 | 4 | 7 | 5 | 6 |
+--------+---+---+---+---+---+---+---+---+
|        | 0 | 3 | 1 | 2 | 5 | 6 | 4 | 7 |
+--------+---+---+---+---+---+---+---+---+
|        | 0 | 3 | 1 | 2 | 5 | 7 | 4 | 6 |
+--------+---+---+---+---+---+---+---+---+
|        | 0 | 3 | 1 | 2 | 6 | 7 | 4 | 5 |
+--------+---+---+---+---+---+---+---+---+
|        | 1 | 2 | 0 | 3 | 4 | 5 | 6 | 7 |
+--------+---+---+---+---+---+---+---+---+
|        | 1 | 2 | 0 | 3 | 4 | 6 | 5 | 7 |
+--------+---+---+---+---+---+---+---+---+
|        | 1 | 2 | 0 | 3 | 4 | 7 | 5 | 6 |
+--------+---+---+---+---+---+---+---+---+
|        | 1 | 2 | 0 | 3 | 5 | 6 | 4 | 7 |
+--------+---+---+---+---+---+---+---+---+
|        | 1 | 2 | 0 | 3 | 5 | 7 | 4 | 6 |
+--------+---+---+---+---+---+---+---+---+
|        | 1 | 2 | 0 | 3 | 6 | 7 | 4 | 5 |
+--------+---+---+---+---+---+---+---+---+
|        | 1 | 3 | 0 | 2 | 4 | 5 | 6 | 7 |
+--------+---+---+---+---+---+---+---+---+
|        | 1 | 3 | 0 | 2 | 4 | 6 | 5 | 7 |
+--------+---+---+---+---+---+---+---+---+
|        | 1 | 3 | 0 | 2 | 4 | 7 | 5 | 6 |
+--------+---+---+---+---+---+---+---+---+
|        | 1 | 3 | 0 | 2 | 5 | 6 | 4 | 7 |
+--------+---+---+---+---+---+---+---+---+
|        | 1 | 3 | 0 | 2 | 5 | 7 | 4 | 6 |
+--------+---+---+---+---+---+---+---+---+
|        | 1 | 3 | 0 | 2 | 6 | 7 | 4 | 5 |
+--------+---+---+---+---+---+---+---+---+
|        | 2 | 3 | 0 | 1 | 4 | 5 | 6 | 7 |
+--------+---+---+---+---+---+---+---+---+
|        | 2 | 3 | 0 | 1 | 4 | 6 | 5 | 7 |
+--------+---+---+---+---+---+---+---+---+
|        | 2 | 3 | 0 | 1 | 4 | 7 | 5 | 6 |
+--------+---+---+---+---+---+---+---+---+
|        | 2 | 3 | 0 | 1 | 5 | 6 | 4 | 7 |
+--------+---+---+---+---+---+---+---+---+
|        | 2 | 3 | 0 | 1 | 5 | 7 | 4 | 6 |
+--------+---+---+---+---+---+---+---+---+
|        | 2 | 3 | 0 | 1 | 6 | 7 | 4 | 5 |
+--------+---+---+---+---+---+---+---+---+

Bootstrapping raw values
^^^^^^^^^^^^^^^^^^^^^^^^

This method is similar to the permutation test, but rather than create
R permutations, we create R *legal* lists of indexes by sampling the
indexes from each group defined by the reduced model (nuisance
variables) with replacement. So in this case the same column may
appear more than once in a sample, and there may be redundant samples.

For example, we might have:

+--------+---------------+---------------+
| Sex    | Male          | Female        |
+--------+-------+-------+-------+-------+
|Treated | No    | Yes   | No    | Yes   |
+========+===+===+===+===+===+===+===+===+
|        | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
+--------+---+---+---+---+---+---+---+---+
|        | 1 | 0 | 2 | 3 | 4 | 5 | 6 | 7 |
+--------+---+---+---+---+---+---+---+---+
|        | 0 | 0 | 0 | 0 | 4 | 4 | 4 | 4 |
+--------+---+---+---+---+---+---+---+---+

Note that we still restrict the columns by the reduced layout.

Bootstrapping residuals
^^^^^^^^^^^^^^^^^^^^^^^

