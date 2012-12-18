import contextlib
import tempfile
import shutil
import page.main
import os

@contextlib.contextmanager
def tempdir():
    d = tempfile.mkdtemp()
    try:
        yield d
    finally:
        shutil.rmtree(d)

@contextlib.contextmanager
def sample_job(infile, factor_map):
    schema = page.main.init_schema(infile)

    with tempdir() as tmp:
        filename = os.path.join(tmp, 'schema.yaml')
        job = page.main.init_job(
            directory=tmp,
            infile=infile,
            factors=factor_map.keys())
        schema = job.schema
        
        for factor, col_to_val in factor_map.items():
            for col, val in col_to_val.items():
                schema.set_factor(col, factor, val)
                
        with open(job.schema_filename, 'w') as out:
            schema.save(out)

        yield job
    
