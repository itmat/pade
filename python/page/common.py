"""Some generally useful functions.

We may want to pull these out of PaGE someday and put them in a shared
project.

"""
import os
import logging
import contextlib
import resource
import numpy as np
import textwrap

def double_sum(data):
    """Returns the sum of data over the first two axes."""
    return np.sum(np.sum(data, axis=-1), axis=-1)


@contextlib.contextmanager
def chdir(path):
    """Context manager for changing working directory.

    cds to the given path, yeilds, then changes back.

    >>> with chdir("/tmp"):
    >>>   # Do something in the tmp dir
    >>>   pass
    
    """
    cwd = os.getcwd()
    try:
        logging.debug("Changing cwd from " + cwd + " to " + path)
        os.chdir(path)
        yield
    finally:
        logging.debug("Changing cwd from " + path + " back to " + cwd)
        os.chdir(cwd)


@contextlib.contextmanager
def figure(path):
    """Context manager for saving a figure.

    Clears the current figure, yeilds, then saves the figure to the
    given path and clears the figure again.

    """
    try:
        logging.debug("Creating figure " + path)
        plt.clf()
        yield
        plt.savefig(path)
    finally:
        plt.clf()


def fix_newlines(msg):
    """Attempt to wrap long lines as paragraphs.

    Treats each line in the given message as a paragraph. Wraps each
    paragraph to avoid long lines. Returns a string that contains all
    the wrapped paragraphs, separated by blank lines.

    """
    output = ""
    for line in msg.splitlines():
        output += textwrap.fill(line) + "\n"
    return output


def maxrss():
    """Return the current maximum RSS allocation in GB."""
    bytes = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return bytes / 1000000000.0
    

