from __future__ import print_function

import sys, os

bin_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
pade_root = os.path.dirname(bin_dir)

print("In development mode; prepending", pade_root, "to sys.path")

sys.path.insert(0, pade_root)
