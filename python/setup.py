from distutils.core import setup

setup(name='PaGE',
      version='6.0',
      author='Mike DeLaurentis',
      author_email='delaurentis@gmail.com',
      url='https://github.com/itmat/PaGE',
      scripts=['bin/page'],
      packages=['page'],
      requires=['jinja2', 'matplotlib', 'scipy', 'numpy', 'yaml'],
      data_files=['page/thirdparty/996grid/code/css/*.css'])
