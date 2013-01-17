import numbers
import numpy as np
import numpy.ma as ma
from page.common import *


def apply_layout(layout, data):
    """Reshape the given data so based on the layout.

    layout - A list of lists. Each inner list is a sequence of indexes
    into data, representing a group of samples that share the same
    factor values. All of the inner lists must currently have the same
    length.

    data - An m x n array, where m is the number of samples and n is
    the number of features.

    Returns an m1 x m2 x n array, where m1 is the number of groups, m2
    is the number of samples in each group, and n is the number of features.

    """
    shape = np.shape(data)[:-1] + (len(layout), len(layout[0]))

    res = np.zeros(shape)

    for i, idxs in enumerate(layout):
        res[..., i, :] = data[..., idxs]
    return res

def mean_and_rss(data):
    y   = np.mean(data, axis=-1)
    rss = double_sum((data  - y)  ** 2)
    return (y, rss)

class Ftest:

    def __init__(self, layout_full, layout_reduced, alphas=None):
        self.layout_full = layout_full
        self.layout_reduced = layout_reduced
        self.alphas = alphas

    def __call__(self, data):
        """Compute the f-test for the given ndarray.

        Input must have 2 or more dimensions. Axis 0 must be sample,
        axis 1 must be condition. Operations are vectorized over any
        subsequent axes. So, for example, an input array with shape
        (3, 2) would represent 1 feature for 2 conditions, each with
        at most 3 samples. An input array with shape (5, 3, 2) would
        be 5 features for 3 samples of 2 conditions.

        TODO: Make sure masked input arrays work.

        """
        data_full = apply_layout(self.layout_full, data)
        data_red  = apply_layout(self.layout_reduced,  data)

        # Means for the full and reduced model
        y_full = np.mean(data_full, axis=-1)
        y_red  = np.mean(data_red,  axis=-1)
        y_full = y_full.reshape(np.shape(data_full)[:-1] + (1,))
        y_red = y_red.reshape(np.shape(data_red)[:-1] + (1,))

        # Degrees of freedom
        p_red  = len(self.layout_reduced)
        p_full = len(self.layout_full)
        n = len(self.layout_reduced) * len(self.layout_reduced[0])

        # Residual sum of squares for the reduced and full model
        rss_red  = double_sum((data_red  - y_red)  ** 2)
        rss_full = double_sum((data_full - y_full) ** 2)

        numer = (rss_red - rss_full) / (p_full - p_red)
        denom = rss_full / (n - p_full)

        if self.alphas is not None:
            denom = np.array([denom + x for x in self.alphas])
        return numer / denom

class FtestSqrt:
    def __init__(self, layout_full, layout_reduced):
        self.test = Ftest(layout_full, layout_reduced)
        
    def __call__(self, data):
        return np.sqrt(self.test(data))

class Ttest:

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

    @classmethod
    def compute_s(cls, data):
        """
        v1 and v2 should have the same number of rows.
        """
                
        var = np.var(data, ddof=1, axis=1)
        size = ma.count(data, axis=1)
        return np.sqrt(np.sum(var * size, axis=0) / np.sum(size, axis=0))


    @classmethod
    def find_default_alpha(cls, table):
        """
        Return a default value for alpha. 
        
        Table should be an ndarray, with shape (conditions, samples, features).
        
        """

        alphas = np.zeros(len(table))
        (num_classes, samples_per_class, num_features) = np.shape(table)

        for c in range(1, num_classes):
            subset = table[([c, 0],)]
            values = cls.compute_s(subset)
            mean = np.mean(values)
            residuals = values[values < mean] - mean
            sd = np.sqrt(sum(residuals ** 2) / (len(residuals) - 1))
            alphas[c] = mean * 2 / np.sqrt(samples_per_class * 2)

        return alphas


    def __call__(self, data):
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




