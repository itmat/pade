import logging
import os.path
import yaml

path = [
    os.path.expanduser('~/.padeconfig.yaml'),
    '/etc/padeconfig.yaml'
    ]

def search():
    logging.info("Searching for pade config")
    for p in path:
        try:
            return load(p)
        except IOError as e:
            logging.info("Couldn't load config file from " + str(p))

    raise Exception("I couldn't find a configuration file to load. I tried " +
                    "these paths: " + str(path))

def load(path):
    logging.info("Trying to load PADE config from " + path)
    with open(path) as f:
        doc = yaml.load(f)

        return PadeConfig(
            metadb_host=doc['metadb_host'],
            metadb_port=doc['metadb_port'],
            metadb_db=doc['metadb_db'],

            session_host=doc['session_host'],
            session_port=doc['session_port'],
            session_db=doc['session_db'],

            celery_host=doc['celery_host'],
            celery_port=doc['celery_port'],
            celery_db=doc['celery_db'],

            metadb_dir=doc['metadb_dir'])
            

class PadeConfig(object):

    def __init__(self, 
             metadb_host=None,
             metadb_port=None,
             metadb_db=None,

             session_host=None,
             session_port=None,
             session_db=None,

             celery_host=None,
             celery_port=None,
             celery_db=None,

             metadb_dir=None):


        self.metadb_host = metadb_host
        self.metadb_port = metadb_port
        self.metadb_db   = metadb_db

        self.session_host = session_host
        self.session_port = session_port
        self.session_db   = session_db

        self.celery_host = celery_host
        self.celery_port = celery_port
        self.celery_db   = celery_db

        self.metadb_dir = metadb_dir

default = PadeConfig(
    metadb_host='localhost',
    metadb_port=6379,
    metadb_db=0,

    session_host='localhost',
    session_port=6379,
    session_db=0,

    celery_host='localhost',
    celery_port=6379,
    celery_db=0,

    metadb_dir='padedb')

test = PadeConfig(
    metadb_host='localhost',
    metadb_port=6379,
    metadb_db=1,

    session_host='localhost',
    session_port=6379,
    session_db=1,

    celery_host='localhost',
    celery_port=6379,
    celery_db=1,

    metadb_dir='padedb_test')
