"""Utilities for measuring performance of PaGE.

"""

import contextlib
import resource
import numpy as np

EVENTS = []

@contextlib.contextmanager
def profiling(label):
    """Context manager for capturing performance metrics of a block.

    """
    pre_maxrss = maxrss()
    EVENTS.append(('enter', label, maxrss()))
    yield
    post_maxrss = maxrss()
    EVENTS.append(('exit', label, maxrss()))

def profiled(method):
    """Decorator for profiling a function."""
    def wrapped(*args, **kw):
        with profiling(method.__name__):
            return method(*args, **kw)

    return wrapped

def maxrss():
    """Return the current maximum RSS allocation in GB."""
    bytes = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return bytes / 1000000000.0

def walk_profile(profile_log=EVENTS):
    """Analyze the given list of events and return a table."""
    stack = []
    events = []
    order = 0
    for entry in profile_log:
        (event, label, maxrss) = entry
        if event == 'enter':
            stack.append({
                    'order'      : order,
                    'depth'      : len(stack),
                    'label'      : label,
                    'maxrss_pre' : maxrss })
            order += 1
        elif event == 'exit':
            entry = stack.pop()
            if entry['label'] != label:
                raise ProfileStackException(
                    "Expected to pop {0}, got {1} instead".format(
                        label, entry['label']))
            entry['maxrss_post'] = maxrss
            events.append(entry)
        else:
            raise ProfileStackException(
                "Unknown event " + event)

    recs = [(e['depth'], 
             e['order'],
             e['label'],
             e['maxrss_pre'], 
             e['maxrss_post'],
             0.0,
             0.0)
            for e in events]

    dtype=[('depth', int),
           ('order', int),
           ('label', 'S100'),
           ('maxrss_pre', float),
           ('maxrss_post', float),
           ('maxrss_diff', float),
           ('maxrss_diff_percent', float)]

    table = np.array(recs, dtype)

    table['maxrss_diff'] = table['maxrss_post'] - table['maxrss_pre']
    
    table['maxrss_diff_percent'] = table['maxrss_diff']
    if len(table['maxrss_post']) > 0:
        table['maxrss_diff_percent'] /= max(table['maxrss_post'])
    table.sort(order=['order'])
    return table

if __name__ == '__main__':
    import doctest
    doctest.testmod()

