import contextlib
import datetime
import redisconfig
import unittest

from StringIO import StringIO
from pade.test.utils import tempdir
from pade.model import Schema
from pade.metadb import *
from redis import Redis


@contextlib.contextmanager
def temp_metadb():
    with tempdir() as d:
        mdb = MetaDB(d, Redis(db=redisconfig.DB_METADB_TEST))
        mdb.redis.flushdb()
        yield mdb

class MetaDBTest(unittest.TestCase):

    def test_next_obj_id(self):
        with temp_metadb() as mdb:
            a = mdb._next_obj_id('input_file')
            b = mdb._next_obj_id('input_file')
            self.assertGreater(b, a)

    def test_input_files(self):
        with temp_metadb() as mdb:

            a = mdb.add_input_file(name="test.txt", description="Some comments", stream=StringIO("a\nb\nc\n"))
            b = mdb.add_input_file(name="foo.txt", description="Other comments", stream=StringIO("1\n2\n3\n"))

            # Make sure it returned the object appropriately
            self.assertEquals(a.name, "test.txt")
            self.assertEquals(a.description, "Some comments")
            with open(a.path) as f:
                self.assertEquals("a\nb\nc\n", f.read())

            self.assertEquals(type(a.dt_created), datetime.datetime)

            # Make sure we can list all input files
            input_files = mdb.all_input_files()
            self.assertEquals(len(input_files), 2)
            names = set(['test.txt', 'foo.txt'])
            self.assertEquals(names, set([x.name for x in input_files]))

    def test_schemas(self):
        with temp_metadb() as mdb:

            rawfile = mdb.add_input_file(name="test.txt", description="Some comments", stream=StringIO("a\nb\nc\n"))

            schema_a = Schema()
            schema_a.add_factor('treated', [False, True])
            schema_a.set_columns(['id', 'a', 'b'],
                                 ['feature_id', 'sample', 'sample'])
            schema_a.set_factor('a', 'treated', False)
            schema_a.set_factor('b', 'treated', True)

            schema_b = Schema()
            schema_b.add_factor('age', ['young', 'old'])
            schema_b.set_columns(['key', 'foo', 'bar'],
                                 ['feature_id', 'sample', 'sample'])
            schema_b.set_factor('foo', 'age', 'young')
            schema_b.set_factor('bar', 'age', 'old')

            a = mdb.add_schema("First one", "The first one", schema_a, rawfile)
            b = mdb.add_schema("Second", "Other", schema_b, rawfile)
            
            self.assertEquals(a.name, "First one")
            self.assertEquals(a.description, "The first one")
            
            schemas = mdb.all_schemas()
            self.assertEquals(len(schemas), 2)

            self.assertEquals(a.based_on_input_file_id, 
                              rawfile.obj_id)

            colnames = set()
            for s in schemas:
                schema = s.load()
                colnames.update(schema.column_names)
            self.assertEquals(colnames, set(['id', 'a', 'b',
                                             'key', 'foo', 'bar']))

            schema_ids = mdb.schemas_based_on_input_file(a.based_on_input_file_id)
            self.assertTrue(a.obj_id in schema_ids)
            self.assertTrue(b.obj_id in schema_ids)

            
    def test_jobs(self):
        with temp_metadb() as mdb:

            # Set up the raw file
            raw_file_meta = mdb.add_input_file(name="test.txt", description="Some comments", stream=StringIO("a\nb\nc\n"))

            schema = Schema()
            schema.add_factor('treated', [False, True])
            schema.set_columns(['id', 'a', 'b'],
                                 ['feature_id', 'sample', 'sample'])
            schema.set_factor('a', 'treated', False)
            schema.set_factor('b', 'treated', True)

            schema_meta = mdb.add_schema("First one", "The first one", schema, raw_file_meta)

            a = mdb.add_job(name="job1", 
                            description="Some job",
                            raw_file_meta=raw_file_meta,
                            schema_meta=schema_meta)
            b = mdb.add_job(name="job2", description="Other job",
                            raw_file_meta=raw_file_meta,
                            schema_meta=schema_meta)

            # Make sure it returned the object appropriately
            self.assertEquals(a.name, "job1")
            self.assertEquals(a.description, "Some job")

            a = mdb.job(a.obj_id)
            self.assertEquals(a.raw_file_id, raw_file_meta.obj_id)
            self.assertEquals(a.schema_id, schema_meta.obj_id)

            # Make sure we can list all input files
            jobs = mdb.all_jobs()
            self.assertEquals(len(jobs), 2)
            names = set(['job1', 'job2'])
            self.assertEquals(names, set([x.name for x in jobs]))

            job_ids = mdb.jobs_for_schema(schema_meta.obj_id)
            self.assertTrue(a.obj_id in job_ids)
            self.assertTrue(b.obj_id in job_ids)

            job_ids = mdb.jobs_for_raw_file(raw_file_meta.obj_id)
            self.assertTrue(a.obj_id in job_ids)
            self.assertTrue(b.obj_id in job_ids)


    def test_import_job(self):
        with temp_metadb() as mdb:

            a = mdb.add_job(name="job1", 
                            description="Some job",
                            stream=StringIO("The job."))
            b = mdb.add_job(name="job2", 
                            description="Other job",
                            stream=StringIO("The other job data."))

            # Make sure it returned the object appropriately
            self.assertEquals(a.name, "job1")
            self.assertEquals(a.description, "Some job")

            # Make sure we can list all input files
            jobs = mdb.all_jobs()
            self.assertEquals(len(jobs), 2)
            names = set(['job1', 'job2'])
            self.assertEquals(names, set([x.name for x in jobs]))

            with open(a.path) as f:
                self.assertEquals(f.next(), "The job.")

if __name__ == '__main__':
    unittest.main()
