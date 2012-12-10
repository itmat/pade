import numpy as np
import numpy.ma as ma
import numbers

class Tstat(object):

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

    #TUNING_PARAM_RANGE_VALUES = np.array([0.5])
    
    #TUNING_PARAM_RANGE_VALUES = np.array([0.0001, 0.5, 10])

    
    def __init__(self, alpha):
        self.alpha = alpha

        if isinstance(alpha, numbers.Number):
            self.children = None
        else:
            self.children = [Tstat(a) for a in alpha]

    def compute(self, data):
        """
        Samples should be on the second to last axis, and class should
        be on the last axis.
        """
        sample_axis = -2
        class_axis  = -1
        n = ma.count(data, axis=sample_axis)
        n1 = n[..., 0]
        n2 = n[..., 1]

        # Variance for each row of v1 and v2 with one degree of
        # freedom. var1 and var2 will be 1-d arrays, one variance for each
        # feature in the input.
        var   = np.var(data, ddof=1, axis=sample_axis)
        means = np.mean(data, axis=sample_axis)

        prod  = var * (n - 1)
        S     = np.sqrt((prod[..., 0] + prod[..., 1]) / (n1 + n2 - 2))
        numer = (means[..., 0] - means[..., 1]) * np.sqrt(n1 * n2)
        denom = (self.alpha + S) * np.sqrt(n1 + n2)

        return numer / denom

class Ftest(object):

    def compute(self, a):

        (num_samples, num_conditions) = np.shape(a)[-2:]
        print "Shape is " + str(np.shape(a))
        print "Got {samples} samples and {conditions} conditions".format(
            samples=num_samples, conditions=num_conditions)

        # TODO: need to allow masked values
        counts = np.array([len(x) for x in np.swapaxes(a, -2, -1)])
        counts = ma.count(a, axis=-2)

        within_group_mean = np.mean(a, axis=-2)

        overall_mean = np.mean(np.mean(a, axis=-1), axis=-1)

        between_group_ss  = np.sum(counts * ((within_group_mean.T - overall_mean) ** 2).T, axis=-1)
        print "Between group ss"
        print between_group_ss
        f_b = float(num_conditions - 1)
        msb = between_group_ss / f_b

        print "Within group mean:"
        print within_group_mean
        tiled_means = np.tile(np.swapaxes(within_group_mean, -1, -2), (np.shape(a)[:-1] + (1, num_samples)))


        centered = np.zeros_like(a)

        for idx in np.ndindex(np.shape(a)):
            group_idx = idx[:-2] + (idx[-1],)
            centered[idx] = a[idx] - within_group_mean[group_idx]

        s_w = np.sum(np.sum(centered ** 2, axis=-1), axis=-1)
        f_w = np.shape(a)[-1] * (np.shape(a)[-2] - 1)
        msw = s_w / f_w

        return msb / msw

class CompositeStat(object):
    def __init__(self, children):
        self.children = children


    
