import unittest
import contextlib
from pade.test.utils import tempdir

from pade.metadb import *
from StringIO import StringIO

@contextlib.contextmanager
def temp_metadb():
    with tempdir() as d:
        mdb = MetaDB(d, 'test')
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

            a = mdb.add_input_file("test.txt", "Some comments", StringIO("a\nb\nc\n"))
            b = mdb.add_input_file("foo.txt", "Other comments", StringIO("1\n2\n3\n"))

            # Make sure it returned the object appropriately
            self.assertEquals(a.name, "test.txt")
            self.assertEquals(a.comments, "Some comments")
            with open(a.path) as f:
                self.assertEquals("a\nb\nc\n", f.read())

            # Make sure we can list all input files
            input_files = mdb.all_input_files()
            self.assertEquals(len(input_files), 2)
            names = set(['test.txt', 'foo.txt'])
            self.assertEquals(names, set([x.name for x in input_files]))
            
if __name__ == '__main__':
    unittest.main()
