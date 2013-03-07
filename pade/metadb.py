import datetime as dt
import os
import shutil

from pade.schema import Schema
from StringIO import StringIO


class MetaDB(object):

    def __init__(self, directory, redis):
        self.directory = directory
        self.redis     = redis

    def _next_obj_id(self, type):
        key = ":".join(('pade', 'nextid', type))
        return self.redis.incr(key)

    def _obj_key(self, obj_type, obj_id):
        return ':'.join(('pade', obj_type, str(obj_id)))

    def _path(self, obj_type, obj_id):
        filename = "{obj_type}_{obj_id}.txt".format(
            obj_type=obj_type,
            obj_id=obj_id)
        return os.path.join(self.directory, filename)

    def _add_obj(self, cls, name, comments, stream=None):

        obj_type = cls.obj_type
        obj_id = self._next_obj_id(obj_type)

        path = self._path(obj_type, obj_id)

        if stream is not None:
            with open(path, 'w') as out:
                shutil.copyfileobj(stream, out)

        meta = cls(obj_id, name, comments, path)
        mapping = {
            'name'       : meta.name,
            'comments'   : meta.comments,
            'dt_created' : str(meta.dt_created) }

        key = self._obj_key(meta.obj_type, meta.obj_id)

        pipe = self.redis.pipeline()
        pipe.hmset(key, mapping)
        pipe.sadd(cls.collection_name, meta.obj_id)
        pipe.execute()        
        return meta

    def _all_objects(self, cls):
        ids = self.redis.smembers(cls.collection_name)
        return [ self._load_obj(cls, obj_id) for obj_id in ids ]

    def _load_obj(self, cls, obj_id):
        key        = self._obj_key(cls.obj_type, obj_id)
        name       = self.redis.hget(key, 'name')
        comments   = self.redis.hget(key, 'comments')
        dt_created = self.redis.hget(key, 'dt_created')
        path       = self._path(cls.obj_type, obj_id)
        return cls(obj_id, name, comments, path, dt_created)

    def add_input_file(self, name, comments, stream):
        return self._add_obj(InputFileMeta, name, comments, stream)

    def all_input_files(self):
        return self._all_objects(InputFileMeta)

    def input_file(self, obj_id):
        return self._load_obj(InputFileMeta, obj_id)

    def add_schema(self, name, comments, schema):
        out = StringIO()
        schema.save(out)
        saved = out.getvalue()
        stream = StringIO(out.getvalue())
        return self._add_obj(SchemaMeta, name, comments, stream)

    def schema(self, obj_id):
        return self._load_obj(SchemaMeta, obj_id)
    
    def all_schemas(self):
        return self._all_objects(SchemaMeta)

    def add_job_db(self, name, comments):
        return self._add_obj(JobDBMeta, name, comments)

    def job_db(self, obj_id):
        return self._load_obj(JobDBMeta, obj_id)

    def all_job_dbs(self):
        return self._all_objects(JobDBMeta)

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

    @property
    def size(self):
        return os.stat(self.path).st_size
        
class JobDBMeta(ObjMeta):
    """Meta-data for an HDF5 job database."""
    obj_type = 'job_db'
    collection_name = 'pade:job_dbs'

class SchemaMeta(ObjMeta):
    """Meta-data for a PADE schema YAML file."""
    obj_type        = 'schema'
    collection_name = 'pade:schemas'

    def load(self):
        with open(self.path) as f:
            return Schema.load(f)
