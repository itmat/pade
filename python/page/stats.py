import numpy as np

class Tstat(object):
    
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

        print "S is " + str(type(S))
        numer  = (np.mean(v1, axis=1) - np.mean(v2, axis=1)) * np.sqrt(n1 * n2)
        denom = (self.alpha + S) * np.sqrt(n1 + n2)

        return numer / denom

