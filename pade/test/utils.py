import contextlib
import tempfile
import shutil
import os

from pade.main import *
from pade.db   import *

@contextlib.contextmanager
def tempdir():
    d = tempfile.mkdtemp()
    try:
        yield d
    finally:
        shutil.rmtree(d)

@contextlib.contextmanager
def sample_db(infile, factor_map):
    schema = init_schema(infile)

    with tempdir() as tmp:
        schema_path = os.path.join(tmp, 'pade_schema.yaml')
        db_path     = os.path.join(tmp, 'pade_db.h5')

        factors = {}
        for f in factor_map:
            values = set(factor_map[f].values())
            factors[f] = values

        schema = init_job(
            schema_path=schema_path,
            infile=infile,
            factors=factors)

        for factor, col_to_val in factor_map.items():
            for col, val in col_to_val.items():
                schema.set_factor(col, factor, val)
                
        with open(schema_path, 'w') as out:
            schema.save(out)

        db = DB(
            schema_path=schema_path,
            schema=schema,
            path=db_path)
        import_table(db, infile)
        yield db
    
