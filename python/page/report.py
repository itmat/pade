import os
import shutil
from jinja2 import Environment, PackageLoader
import contextlib

@contextlib.contextmanager
def chdir(path):
    cwd = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(cwd)


def make_report(job):
    with chdir(job.directory):
        env = Environment(loader=PackageLoader('page'))
        setup_css(env)
        template = env.get_template('index.html')
        with open('index.html', 'w') as out:
            out.write(template.render(
                job=job
                ))

def setup_css(env):
    src = os.path.join(os.path.dirname(__file__),
                       '996grid/code/css')

    shutil.rmtree('css', True)
    shutil.copytree(src, 'css')

    with open('css/custom.css', 'w') as out:
        template = env.get_template('custom.css')
        out.write(template.render())

