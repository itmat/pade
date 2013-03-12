"""Some generally useful functions.

We may want to pull these out of PaGE someday and put them in a shared
project.

"""

import os
import logging
import contextlib

@contextlib.contextmanager
def chdir(path):
    """Context manager for changing working directory.

    cds to the given path, yeilds, then changes back.
    
    """
    cwd = os.getcwd()
    try:
        logging.debug("Changing cwd from " + cwd + " to " + path)
        os.chdir(path)
        yield
    finally:
        logging.debug("Changing cwd from " + path + " back to " + cwd)
        os.chdir(cwd)


def makedirs(path):
    """Attempt to make the directory.

    Attempt to make each director, and raise an exception if it cannot
    be created. Returns normally if the directory was successfully
    created or if it already existed.

    """
    try:
        os.makedirs(path)
    except OSError as e:
        if not os.path.isdir(path):
            raise e

def assert_ndarray(array, name=None, ndim=None):
    if ndim is not None:
        if array.ndim != ndim:
            msg = ("Array argument {name} must be {ndim}-dimensional, " +
                   "but it has shape {shape}")
            raise Exception(msg.format(
                    name=name,
                    ndim=ndim,
                    shape=array.shape))
                

