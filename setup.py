from setuptools import setup

setup(name='pade',
      version='0.2.0a5',
      author='Mike DeLaurentis',
      author_email='midel@upenn.edu',
      url='https://github.com/itmat/pade',
      scripts=['bin/pade'],
      packages=['pade', 'pade.http', 'pade.http.static.css', 'pade.http.templates'],
      install_requires=['jinja2', 'matplotlib', 'scipy', 'numpy', 'redis', 'celery', 'h5py', 'flask', 'Flask-WTF'],
      package_data={
        'pade.http.static.css' : ['*.css'],
        'pade.http.templates' : ['*']}
      )
