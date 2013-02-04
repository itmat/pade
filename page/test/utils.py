import contextlib
import tempfile
import shutil
import os

from page.main import *

@contextlib.contextmanager
def tempdir():
    d = tempfile.mkdtemp()
    try:
        yield d
    finally:
        shutil.rmtree(d)

@contextlib.contextmanager
def sample_job(infile, factor_map):
    schema = init_schema(infile)

    with tempdir() as tmp:
        filename = os.path.join(tmp, 'schema.yaml')

        factors = {}
        for f in factor_map:
            values = set(factor_map[f].values())
            factors[f] = values

        job = init_job(
            directory=tmp,
            infile=infile,
            factors=factors)
        schema = job.schema

        for factor, col_to_val in factor_map.items():
            for col, val in col_to_val.items():
                schema.set_factor(col, factor, val)
                
        with open(job.schema_path, 'w') as out:
            schema.save(out)

        yield job
    
