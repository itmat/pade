# TODO:
#  Pre-generate histograms of stat distributions
from __future__ import absolute_import, print_function, division

import logging 
import pade.redis_session
import pade.http.jobdetails
import pade.http.newjob
import pade.http.inputfile
import pade.http.jobbrowser

from redis import Redis
from flask import Flask
from pade.metadb import MetaDB

class PadeServer(Flask):
    def __init__(self):
        super(PadeServer, self).__init__(__name__)
        self.jinja_env.filters['datetime'] = datetime_format
        self.jinja_env.filters['joblabel'] = job_label

def redis_db(config):
    return Redis(host=config['host'],
                 port=config['port'],
                 db=config['db'])

class PadeRunner(PadeServer):

    def __init__(self, config):
        super(PadeRunner, self).__init__()
        self.secret_key = ""

        self.register_blueprint(pade.http.jobdetails.bp)
        self.register_blueprint(pade.http.newjob.bp,     url_prefix="/new_job")
        self.register_blueprint(pade.http.inputfile.bp,  url_prefix="/input_files")
        self.register_blueprint(pade.http.jobbrowser.runner)

        self.session_interface = pade.redis_session.RedisSessionInterface(
            redis_db(config['redis']['session']))

        mdb = MetaDB(config['metadb_dir'], redis_db(config['redis']['metadb']))

        modules = [pade.http.jobdetails,
                   pade.http.newjob,
                   pade.http.inputfile,
                   pade.http.jobbrowser]

        for m in modules:
            m.mdb = mdb

        self.add_url_rule('/', 'index', pade.http.jobbrowser.index)

class PadeViewer(PadeServer):
    def __init__(self):
        super(PadeViewer, self).__init__()
        pade.http.jobdetails.mdb = None
        self.register_blueprint(pade.http.jobdetails.bp)
        self.add_url_rule('/', 'index', pade.http.jobdetails.job_list)

def datetime_format(dt):
    """Jinja2 filter for formatting datetime objects."""
    
    return "" if dt is None else dt.strftime('%F %R')

def job_label(job_meta):
    if job_meta.name is not None and len(job_meta.name) > 0:
        return job_meta.name
    else:
        return "job " + str(job_meta.obj_id)
