import sys, os

sys.path.insert(0, os.path.dirname(__file__) + "/../..")

import unittest
import doctest
import page

def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(page))
    return tests
