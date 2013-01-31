Installing PaGE
===============

You can obtain PaGE by cloning or forking it from github here:
https://github.com/itmat/PaGE. 

.. NOTE::
   We need to create a tarball and put it somewhere to be downloaded.

Once you've obtained PaGE, you can either run it from the directory
where you've unpacked it, or install it globally. If you run it from a
local directory, you'll need to update your PYTHONPATH environment
variable to point to the root of the PaGE directory. For example, if
you've downloaded PaGE to ~/projects/PaGE, you might want to add ::

  export PYTHONPATH=$PYTHONPATH:$HOME/projects/PaGE

to your ~/.profile or ~/.bashrc file.

You can install it simply by running ``python setup.py install``.  Note
that you may need to run this as root or under sudo.
   
Prerequisites
-------------

PaGE requires a number of libraries that aren't built in to Python:

* jinja2
* matplotlib
* numpy
* scipy
* yaml

If you are using Linux, you may be able to install these packages
quite easily using `pip <http://pypi.python.org/pypi/pip>` or another
package management tool.

