import datetime as dt
import os
import shutil
import logging

from StringIO import StringIO
from pade.model import Schema

DATE_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

class MetaDB(object):

    def __init__(self, directory, redis):
        logging.info("Initializing metadb for directory " +
                     str(directory) + ", redis " + str(redis))
        self.directory = directory
        self.redis     = redis

    def _next_obj_id(self, type):
        key = ":".join(('pade', 'nextid', type))
        return self.redis.incr(key)

    def _obj_key(self, obj_type, obj_id):
        return ':'.join(('pade', obj_type, str(obj_id)))

    def _link_key(self, from_obj_type, to_obj_type, from_obj_id):
        return ':'.join(('pade', str(from_obj_type) + 'To' + str(to_obj_type), str(from_obj_id)))

    def _path(self, obj_type, obj_id):
        filename = "{obj_type}_{obj_id}.txt".format(
            obj_type=obj_type,
            obj_id=obj_id)
        return os.path.join(self.directory, filename)

    def _add_obj(self, cls, stream=None, **kwargs):

        obj_type = cls.obj_type
        obj_id = self._next_obj_id(obj_type)

        path = self._path(obj_type, obj_id)

        if stream is not None:
            with open(path, 'w') as out:
                shutil.copyfileobj(stream, out)

        dt_created = dt.datetime.now()
        kwargs['dt_created'] = dt_created

        meta = cls(obj_id, path=path, **kwargs)

        key = self._obj_key(meta.obj_type, meta.obj_id)

        pipe = self.redis.pipeline()
        pipe.hmset(key, kwargs)
        pipe.sadd(cls.collection_name, meta.obj_id)
        pipe.execute()        
        return meta

    def _all_objects(self, cls):
        ids = self.redis.smembers(cls.collection_name)
        return [ self._load_obj(cls, obj_id) for obj_id in ids ]

    def _load_obj(self, cls, obj_id):
        key    = self._obj_key(cls.obj_type, obj_id)
        kwargs = self.redis.hgetall(key)
        path   = self._path(cls.obj_type, obj_id)
        kwargs['dt_created'] = dt.datetime.strptime(kwargs['dt_created'], DATE_FORMAT)
        return cls(obj_id, path=path, **kwargs)

    def add_input_file(self, name, stream, description=None):
        return self._add_obj(InputFileMeta, name=name, description=description, stream=stream)

    def all_input_files(self):
        return self._all_objects(InputFileMeta)

    def input_file(self, obj_id):
        res = self._load_obj(InputFileMeta, obj_id)
        return res

    def _link_one_to_many(self, one, many, field_name):
        many_key = self._obj_key(many.obj_type, many.obj_id)
        link_key = self._link_key(one.obj_type, many.obj_type, one.obj_id)
        pipe = self.redis.pipeline()
        pipe.hset(many_key, field_name, one.obj_id)
        pipe.sadd(link_key, many.obj_id)
        pipe.execute()
        many.__dict__[field_name] = one.obj_id
        return many

    def add_schema(self, name, description, schema, raw_file_meta):
        out = StringIO()
        schema.save(out)
        saved = out.getvalue()
        stream = StringIO(out.getvalue())

        schema_meta = self._add_obj(SchemaMeta, name=name, description=description, 
                             stream=stream)

        return self._link_one_to_many(raw_file_meta, schema_meta, 'based_on_input_file_id')

    def schema(self, obj_id):
        return self._load_obj(SchemaMeta, obj_id)
    
    def all_schemas(self):
        return self._all_objects(SchemaMeta)

    def add_job(self, name, description):
        job_meta = self._add_obj(JobMeta, name=name, description=description)
        return job_meta

    def job(self, obj_id):
        return self._load_obj(JobMeta, obj_id)

    def all_jobs(self):
        return self._all_objects(JobMeta)

    def add_task_id(self, obj, task_id):
        self.redis.sadd(self.task_ids_key(obj), task_id)

    def task_ids_key(self, obj):
        return self._obj_key('taskids', str(obj.obj_id))

    def get_task_ids(self, obj):
        return self.redis.smembers(self.task_ids_key(obj))
                        
class ObjMeta(object):
    """Meta-data for an object we store on the filesystem."""
    def __init__(self, obj_id, path, dt_created=None):

        self.obj_id = obj_id
        """Unique id of the object (unique for objects of this type)."""

        self.dt_created = dt_created
        """datetime the object was created."""

        self.path = path
        """Path to the file containing the object."""


class InputFileMeta(ObjMeta):
    """Meta-data for an input file."""
    obj_type        = 'input_file'
    collection_name = 'pade:input_files'
    input_file_to_schemas = 'pade:input_file_to_schemas'

    def __init__(self, obj_id, name, description, path, dt_created=None):
        super(InputFileMeta, self).__init__(obj_id, path, dt_created)

        self.name = name
        """A short, descriptive name for the object."""

        self.description = description
        """Longer, free-form description of the object."""
        
    @property
    def size(self):
        return os.stat(self.path).st_size
        
class JobMeta(ObjMeta):
    """Meta-data for an HDF5 job database."""
    obj_type = 'job'
    collection_name = 'pade:jobs'

    def __init__(self, obj_id, name, path, dt_created=None, description=None):
        super(JobMeta, self).__init__(obj_id, path, dt_created)

        self.name = name
        """A short, descriptive name for the object."""

        self.description = description
        """Longer, free-form description of the object."""

class SchemaMeta(ObjMeta):
    """Meta-data for a PADE schema YAML file."""
    obj_type        = 'schema'
    collection_name = 'pade:schemas'

    def load(self):
        with open(self.path) as f:
            return Schema.load(f)

    def __init__(self, obj_id, name, description, path, dt_created=None, 
                 based_on_input_file_id=None):
        super(SchemaMeta, self).__init__(obj_id, path, dt_created)

        self.name = name
        """A short, descriptive name for the object."""

        self.description = description
        """Longer, free-form description of the object."""

        self.based_on_input_file_id = based_on_input_file_id
