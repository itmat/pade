import numpy as np

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
