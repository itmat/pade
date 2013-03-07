import datetime as dt
import os
import redis
import shutil

from redis import Redis

import shutil

REDIS_DB_NUMBERS = {
    'scratch' : 0,
    'dev'     : 1,
    'test'    : 2,
    'prod'    : 3
}

class MetaDB(object):

    def __init__(self, directory, env):
        self.redis = Redis(db=REDIS_DB_NUMBERS[env])
        self.directory = directory

    def next_obj_id(self, type):
        key = ":".join(('pade', 'nextid', type))
        return self.redis.incr(key)

    def obj_key(self, obj_type, obj_id):
        return ':'.join(('pade', obj_type, str(obj_id)))

    def _add_meta(self, meta):
        mapping = {
            'name'       : meta.name,
            'comments'   : meta.comments,
            'dt_created' : str(meta.dt_created) }

        key    = self.obj_key(meta.obj_type, meta.obj_id)

        pipe = self.redis.pipeline()
        pipe.hmset(key, mapping)
        pipe.sadd(InputFileMeta.collection_name, meta.obj_id)
        pipe.execute()        


    def _path(self, obj_type, obj_id):
        filename = "{obj_type}_{obj_id}.txt".format(
            obj_type=obj_type,
            obj_id=obj_id)
        return os.path.join(self.directory, filename)

    def add_input_file(self, name, comments, stream):

        obj_type = InputFileMeta.obj_type
        obj_id = self.next_obj_id(obj_type)

        path = self._path(obj_type, obj_id)

        with open(path, 'w') as out:
            shutil.copyfileobj(stream, out)

        meta = InputFileMeta(obj_id, name, comments, path)
        self._add_meta(meta)
        return meta

    def all_input_files(self):
        ids = self.redis.smembers('pade:input_files')
        return [ self.input_file(obj_id) for obj_id in ids ]

    def input_file(self, obj_id):
        key        = self.obj_key(InputFileMeta.obj_type, obj_id)
        name       = self.redis.hget(key, 'name')
        comments   = self.redis.hget(key, 'comments')
        dt_created = self.redis.hget(key, 'dt_created')
        path       = self._path(InputFileMeta.obj_type, obj_id)
        return InputFileMeta(obj_id, name, comments, path, dt_created)

class ObjMeta(object):
    """Meta-data for an object we store on the filesystem."""
    def __init__(self, obj_id, name, comments, path, dt_created=dt.datetime.now()):

        self.obj_id = obj_id
        """Unique id of the object (unique for objects of this type)."""

        self.name = name
        """A short, descriptive name for the object."""

        self.comments = comments
        """Longer, free-form description of the object."""
        
        self.path = path
        """Path to the file containing the object."""

        self.dt_created = dt_created
        """datetime the object was created."""

class InputFileMeta(ObjMeta):
    """Meta-data for an input file."""
    obj_type        = 'input_file'
    collection_name = 'pade:input_files'
    
class JobDBMeta(ObjMeta):
    """Meta-data for an HDF5 job database."""
    obj_type = 'job_db'
    collection_name = 'pade:job_dbs'

class SchemaMeta(ObjMeta):
    """Meta-data for a PADE schema YAML file."""
    obj_type        = 'schema'
    collection_name = 'pade:schemas'

