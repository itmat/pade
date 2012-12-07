import numpy as np
import numpy.ma as ma

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

    def compute(self, (v1, v2)):
        # n1 and n2 are the length of each row. TODO: When we start using
        # masked values we will need to use the number of unmasked values
        # in each row. Until then, all the lengths are the same.
        m = len(v1)
        n1 = np.array([len(row) for row in v1])
        n2 = np.array([len(row) for row in v2])

        # Variance for each row of v1 and v2 with one degree of
        # freedom. var1 and var2 will be 1-d arrays, one variance for each
        # feature in the input.
        var1 = np.var(v1, ddof=1, axis=1)
        var2 = np.var(v2, ddof=1, axis=1)

        S = np.sqrt((var1 * (n1-1) + var2 * (n2-1)) /(n1 + n2 - 2))

        numer  = (np.mean(v1, axis=1) - np.mean(v2, axis=1)) * np.sqrt(n1 * n2)
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

        # TODO: Not nd friendly

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


    
