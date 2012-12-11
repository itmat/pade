import numpy as np
import numpy.ma as ma
import numbers

class Ttest(object):

    TUNING_PARAM_RANGE_VALUES = np.array([
            0.0001,
            0.01,
            0.1,
            0.3,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            10,
            ])

    
    def __init__(self, alpha):
        self.alpha = alpha

        if isinstance(alpha, numbers.Number):
            self.children = None
        else:
            self.children = [Ttest(a) for a in alpha]


    def compute(self, data):
        """Computes the t-stat.

        Input must be an ndarray with at least 2 dimensions. Axis 0
        should be class, and axis 1 should be sample. If there are
        more than two axes, the t-stat will be vectorized to all axes
        past axis .
        """

        class_axis = 0
        sample_axis = 1

        n = ma.count(data, axis=1)
        n1 = n[0]
        n2 = n[1]

        # Variance for each row of v1 and v2 with one degree of
        # freedom. var1 and var2 will be 1-d arrays, one variance for each
        # feature in the input.
        var   = np.var(data, ddof=1, axis=sample_axis)
        means = np.mean(data, axis=sample_axis)
        prod  = var * (n - 1)
        S     = np.sqrt((prod[0] + prod[1]) / (n1 + n2 - 2))
        numer = (means[0] - means[1]) * np.sqrt(n1 * n2)
        denom = (self.alpha + S) * np.sqrt(n1 + n2)

        return numer / denom


class Ftest(object):

    def compute(self, a):
        """Compute the f-test for the given ndarray.

        Input must be have 2 or more dimensions. Axis 0 must be
        sample, axis 1 must be class (or condition). Operations are
        vectorized over any subsequent axes. So, for example, an input
        array with shape (3, 2) would represent 1 feature for 2
        classes, each with at most 3 samples. An input array with
        shape (5, 3, 2) would be 5 features for 3 samples of 2
        conditions.

        TODO: Make sure masked input arrays work.

        """
        (num_samples, num_conditions) = np.shape(a)[:2]

        # u_w is within-group mean (over axis 0, the sample axis)
        # u is overall mean (mean of the within-group means)
        # s_b is between-group sum of squares
        # s_w is within-group sum of squares
        # msb and msw are mean-square values for between-group and
        # within-group
        u_w = np.mean(a, axis=0)    # Within-group mean
        u   = np.mean(u_w, axis=0) # Overall mean
        s_b = np.sum(ma.count(a, axis=0) * (u_w - u) ** 2, axis=0)
        s_w = np.sum(np.sum((a - u_w) ** 2, axis=0), axis=0)
        msb = s_b / float(num_conditions - 1)
        msw = s_w / (num_conditions * (num_samples - 1))

        return msb / msw


    
