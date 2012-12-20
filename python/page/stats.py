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

def double_sum(data):
    return np.sum(np.sum(data, axis=0), axis=0)

class Ftest(object):

    def __init__(self, full_model, reduced_model):
        self.full_model = full_model
        self.reduced_model = reduced_model
                 

    def compute(self, a):
        """Compute the f-test for the given ndarray.

        Input must have 2 or more dimensions. Axis 0 must be sample,
        axis 1 must be condition. Operations are vectorized over any
        subsequent axes. So, for example, an input array with shape
        (3, 2) would represent 1 feature for 2 conditions, each with
        at most 3 samples. An input array with shape (5, 3, 2) would
        be 5 features for 3 samples of 2 conditions.

        TODO: Make sure masked input arrays work.

        """
        a = np.swapaxes(a, 0, 1)
        (num_samples, num_conditions) = np.shape(a)[:2]

        # Means for the full and reduced model
        y_full = np.mean(a, axis=0)
        y_red   = np.mean(y_full, axis=0)

        # Degrees of freedom
        p_red = 1
        p_full = num_conditions
        n = num_samples * num_conditions
        
        # Residual sum of squares for the reduced and full model
        rss_red  = double_sum((a - y_red)  ** 2)
        rss_full = double_sum((a - y_full) ** 2)

        numer = (rss_red - rss_full) / (p_full - p_red)
        denom = rss_full / (n - p_full)
        return numer / denom

