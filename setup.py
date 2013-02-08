from distutils.core import setup

setup(name='pade',
      version='0.1',
      author='Mike DeLaurentis',
      author_email='midel@upenn.edu',
      url='https://github.com/itmat/pade',
      scripts=['bin/pade'],
      packages=['pade', 'pade.static.css', 'pade.templates'],
      requires=['jinja2', 'matplotlib', 'scipy', 'numpy', 'yaml'],
      package_data={
        'pade.static.css' : ['*.css'],
        'pade.templates' : ['*']}
      )
