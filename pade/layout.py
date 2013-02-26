import numpy as np
from itertools import combinations, product
from scipy.misc import comb

def intersect_layouts(layout0, layout1):

    """Return a layout where each group is the intersection of a group in
    layout0 with a group in layout1.

    >>> block_layout = [[0, 1, 2, 3], [4, 5, 6, 7]]
    >>> condition_layout = [[0, 1, 4, 5], [2, 3, 6, 7]]    
    >>> intersect_layouts(block_layout, condition_layout)
    [[0, 1], [2, 3], [4, 5], [6, 7]]

    """
    layout = []
    for a in map(set, layout0):
        for b in map(set, layout1):
            c = a.intersection(b)
            if len(c) > 0:
                layout.append(list(c))
    return layout
    

def apply_layout(data, layout):
    """Splits data into groups based on layout.

    1d data:
    
    >>> data = np.array([9, 8, 7, 6])
    >>> layout = [ [0, 1], [2, 3] ]
    >>> apply_layout(data, layout) # doctest: +NORMALIZE_WHITESPACE
    [array([9, 8]), array([7, 6])]

    2d data:
    
    >>> data = np.array([[9, 8, 7, 6], [5, 4, 3, 2]])
    >>> layout = [ [0, 1], [2, 3] ]
    >>> apply_layout(data, layout) # doctest: +NORMALIZE_WHITESPACE
    [array([[9, 8], [5, 4]]), array([[7, 6], [3, 2]])]

    """
    return [ data[..., list(idxs)] for idxs in layout]

def layout_is_paired(layout):
    """Returns true of the layout appears to be 'paired'.

    A paired layout is one where each group contains two values.

    :param layout:
       A :term:`layout`      

    :return:
      Boolean indicating if layout appears to be paired.

    """
    for grp in layout:
        if len(grp) != 2:
            return False
    return True

def num_orderings(condition_layout, block_layout=None):
    """Return the number of orderings for the pair of layouts.

    Returns the number of distinct permutations of the indexes that
    are valid within the given block layout and result in distinct
    labelling of samples with conditions.

    :param condition_layout: 
      Layout grouping the samples by condition.

    :param block_layout:
      Optional layout that groups the samples together by blocking
      variables.

      """
    # If there is no block layout, just find the number of
    # orderings of indexes in the full layout.
    if block_layout is None or len(block_layout) == 0:

        # If we only have one group in the full layout, there's only
        # one ordering of the indexes in that group.
        if len(condition_layout) <= 1:
            return 1

        # Otherwise say N is the total number of items in the full
        # layout and k is the number in the 0th group of the full
        # layout. The number of orderings is (N choose k) times the
        # number of orderings for the rest of the groups.
        N = sum(map(len, condition_layout))
        k   = len(condition_layout[0])
        return comb(N, k) * num_orderings(condition_layout[1:])

    # Since we got a block layout, we need to find the number of
    # orderings *within* the first group in the block layout, then
    # multiply that by the orderings in the rest of the block_layout
    # layout. First find the number of groups in the full layout that
    # correspond to the first group in the block_layout layout.

    # First find the number of groups in the full layout that fit in
    # the first group of the block_layout layout.

    prefix = intersect_layouts(condition_layout, block_layout[:1])
    suffix = intersect_layouts(condition_layout, block_layout[1:])

    num_arr_first = num_orderings(prefix)
    num_arr_rest  = num_orderings(suffix, block_layout[1 : ])

    return num_arr_first * num_arr_rest


def all_orderings_within_group(items, sizes):
    """Return all combinations of permutations of the given items within
    groups that have the given sizes.

    One index, one group of size one:

    >>> list(all_orderings_within_group([0], [1]))
    [[0]]

    Two indexes, one group of size two:

    >>> list(all_orderings_within_group([0, 1], [2]))
    [[0, 1]]

    Two indexes, two groups of size one:
    
    >>> list(all_orderings_within_group([0, 1], [1, 1]))
    [[0, 1], [1, 0]]

    >>> list(all_orderings_within_group([0, 1, 2, 3], [2, 2])) # doctest: +NORMALIZE_WHITESPACE
    [[0, 1, 2, 3], 
     [0, 2, 1, 3],
     [0, 3, 1, 2],
     [1, 2, 0, 3],
     [1, 3, 0, 2],
     [2, 3, 0, 1]]
    
    """
    items = set(items)
    if len(items) != sum(sizes):
        raise InvalidLayoutException("Layout is bad")

    for c in map(list, combinations(items, sizes[0])):
        if len(sizes) == 1:
            yield c
        else:
            for arr in all_orderings_within_group(
                items.difference(c), sizes[1:]):
                yield c + arr

def all_orderings(condition_layout, block_layout):
    """Return all valid orderings based on the given layouts.

    :param condition_layout: 
      Layout grouping the samples by condition.

    :param block_layout:
      Optional layout that groups the samples together by blocking
      variables.

    Each ordering returned has a distinct assignment of the indexes
    into the groups defined by condition_layout. In addition each
    ordering preserves the grouping defined by block layout.
      
    """
    grouped = []
    for i, block in enumerate(block_layout):

        cond_groups = intersect_layouts([ block ], condition_layout )
        sizes = map(len, cond_groups)
        grouped.append(all_orderings_within_group(set(block), sizes))

    for prod in product(*grouped):
        row = []
        for block in prod:
            row.extend(block)
        yield row

def random_ordering(layout):
    """Return a randomized ordering of the indexes within each group of
    the given layout.

    """
    row = []
    for grp in layout:
        grp = np.copy(grp)
        np.random.shuffle(grp)
        row.extend(grp)
    return row

def random_orderings(condition_layout, block_layout, R):
    """Get an iterator over at most R random index shuffles.

    :param full: the :term:`layout`
    :param reduced: the reduced :term:`layout`
    :param R: the maximum number of orderings to return

    :return: iterator over random orderings of indexes

    Each item in the resulting iterator will be an ndarray of the
    indexes in the given layouts. The indexes within each group of the
    reduced layout will be shuffled.
    
    """
    full = intersect_layouts(block_layout, condition_layout)

    # Set of random orderings we've returned so far
    orderings = set()
    
    # The total number of orderings of indexes within the groups of
    # the reduced layout that result in a distinct assignment of
    # indexes into the groups defined by the full layout.
    N = num_orderings(condition_layout, block_layout)
    
    # If the number of orderings requested is greater than the number
    # of distinct orderings that actually exist, just return all of
    # them.
    if R >= N:
        for arr in all_orderings(condition_layout, block_layout):
            yield arr

    # Otherwise repeatedly find a random ordering, and if it's not one
    # we've already yielded, yield it.
    else:
        while len(orderings) < R:

            arr = random_ordering(block_layout)
            key = tuple(arr)

            if key not in orderings:
                orderings.add(key)
                yield arr

def random_indexes(layout, R):
    """Generates R samplings of indexes based on the given layout.

    >>> indexes = random_indexes([[0, 1], [2, 3]], 10)
    >>> np.shape(indexes)
    (10, 4)

    """
    layout = [ np.array(grp, int) for grp in layout ]
    n = sum([ len(grp) for grp in layout ])
    res = np.zeros((R, n), int)
    
    for i in range(R):
        p = 0
        q = 0
        for j, grp in enumerate(layout):
            nj = len(grp)
            q  = p + nj
            res[i, p : q] = grp[np.random.random_integers(0, nj - 1, nj)]
            p = q

    return res

