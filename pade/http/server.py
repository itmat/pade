# TODO:
#  Pre-generate histograms of stat distributions
from __future__ import absolute_import, print_function, division

import logging 

import pade.redis_session
import padeconfig

from redis import Redis

from flask import (
    Flask, render_template, make_response, request, session, redirect, 
    url_for, Blueprint)

from pade.stat import cumulative_hist, adjust_num_diff
from pade.metadb import MetaDB

import pade.http.jobdetails
import pade.http.newjob
import pade.http.inputfile
import pade.http.jobbrowser

class PadeApp(Flask):

    def __init__(self):
        super(PadeApp, self).__init__(__name__)
        self.secret_key = ""

app = PadeApp()
app.register_blueprint(pade.http.jobdetails.bp, url_prefix="/jobs/<job_id>/")
app.register_blueprint(pade.http.newjob.bp,     url_prefix="/new_job/")
app.register_blueprint(pade.http.inputfile.bp,  url_prefix="/input_files/")
app.register_blueprint(pade.http.jobbrowser.bp)

app.session_interface = pade.redis_session.RedisSessionInterface(
    Redis(db=padeconfig.DB_SESSION))

mdb = MetaDB(padeconfig.METADB_DIR, Redis(db=padeconfig.DB_METADB))
pade.http.jobdetails.mdb = mdb
pade.http.newjob.mdb = mdb
pade.http.inputfile.mdb = mdb
pade.http.jobbrowser.mdb = mdb

def datetime_format(dt):
    """Jinja2 filter for formatting datetime objects."""

    return dt.strftime('%F %R')
    
app.jinja_env.filters['datetime'] = datetime_format

print("Url map:")
print(app.url_map)
