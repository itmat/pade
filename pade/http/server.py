# TODO:
#  Pre-generate histograms of stat distributions
from __future__ import absolute_import, print_function, division

import logging 
import pade.redis_session
import padeconfig
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


class PadeRunner(PadeServer):

    def __init__(self):
        super(PadeRunner, self).__init__()
        self.secret_key = ""

        self.register_blueprint(pade.http.jobdetails.bp)
        self.register_blueprint(pade.http.newjob.bp,     url_prefix="/new_job")
        self.register_blueprint(pade.http.inputfile.bp,  url_prefix="/input_files")
        self.register_blueprint(pade.http.jobbrowser.runner)

        self.session_interface = pade.redis_session.RedisSessionInterface(
            Redis(db=padeconfig.DB_SESSION))

        mdb = MetaDB(padeconfig.METADB_DIR, Redis(db=padeconfig.DB_METADB))

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
        self.register_blueprint(pade.http.jobdetails.bp)
        self.add_url_rule('/', 'index', pade.http.jobdetails.job_list)

def datetime_format(dt):
    """Jinja2 filter for formatting datetime objects."""
    
    return "" if dt is None else dt.strftime('%F %R')
    
