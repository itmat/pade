# External imports
import logging
import argparse
import os
import errno
import shutil
import numpy as np

from textwrap import fill

# PaGE imports
from schema import Schema
from common import Model, add_condition_axis
import report
import stats

class UsageException(Exception):
    pass


def fix_newlines(msg):
    output = ""
    for line in msg.splitlines():
        output += fill(line) + "\n"
    return output

class Job:

    DEFAULT_DIRECTORY = "pageseq_out"

    def __init__(self, 
                 directory, 
                 stat=None,
                 model=None):
        self.directory = directory
        self.stat_name = stat
        self.model = model
        self._table = None
        self._feature_ids = None

    @property
    def schema_filename(self):
        return os.path.join(self.directory, 'schema.yaml')

    @property
    def input_filename(self):
        return os.path.join(self.directory, 'input.txt')
    
    @property
    def schema(self):
        return Schema.load(open(self.schema_filename), self.input_filename)

    @property
    def stat(self):
        if self.stat_name == 'f':
            return stats.Ftest(None, None)
        elif self.stat_name == 't':
            return stats.Ttest(alpha=1.0)

    @property
    def table(self):
        """Returns the data table as (sample x feature) ndarray."""
        if self._table is None:
            self._load_table()
        return self._table

    @property
    def feature_ids(self):
        if self._feature_ids is None:
            self._load_table()
        return self._feature_ids


    def _load_table(self):
        logging.info("Loading table from " + self.input_filename)
        with open(self.input_filename) as fh:

            headers = fh.next().rstrip().split("\t")

            ids = []
            table = []

            for line in fh:
                row = line.rstrip().split("\t")
                rowid = row[0]
                values = [float(x) for x in row[1:]]
                ids.append(rowid)
                table.append(values)

            self._table = np.array(table).swapaxes(0, 1)
            self._feature_ids = ids

            logging.info("Table shape is " + str(np.shape(self._table)))
            logging.info("Loaded " + str(len(self._feature_ids)) + " features")


    @property
    def num_features(self):
        return len(self.feature_ids)
            

def main():
    """Run pageseq."""

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='page.log',
                        filemode='w')

    args = get_arguments()

    console = logging.StreamHandler()
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)

    if args.debug:
        console.setLevel(logging.DEBUG)
    elif args.verbose:
        console.setLevel(logging.INFO)
    else:
        console.setLevel(logging.ERROR)

    logging.getLogger('').addHandler(console)

    logging.info('Page starting')

    try:
        args.func(args)
    except UsageException as e:
        print fix_newlines("ERROR: " + e.message)
    
    logging.info('Page finishing')    

def do_run(args):

    job = Job(
        directory=args.directory,
        stat=args.stat)

    model = args.model

    if model is None:
        if len(job.schema.factors) == 1:
            model = job.schema.factors[0]
        else:
            msg = """You need to specify a model with --model, since you have more than one factor ({0})."""
            raise UsageException(msg.format(job.schema.factors.keys()))

    schema = job.schema
    model = Model.parse(args.model)

    job.model = model

    if len(model.variables) > 1:
        raise UsageException("I only support models with one factor at this time")
    factor = model.variables[0]
    if factor not in schema.factors:
        raise UsageException("Factor '" + factor + "' is not defined in the schema. Valid factors are " + str(schema.factors.keys()))
    
    groups = schema.sample_groups(factor)
    layout = groups.values()
    group_keys = groups.keys()

    job.condition_names = ["{0}: {1}".format(x[0], x[1]) for x in group_keys]
    logging.debug("Condition names are " + str(job.condition_names))
    logging.debug("Layout is " + str(layout))
    reshaped = add_condition_axis(job.table, layout)
    
    logging.debug("Shape of reshaped is " + str(np.shape(reshaped)))
    job.means = np.mean(reshaped, axis=1)
    logging.debug("Shape of means is " + str(np.shape(job.means)))

    job.stats = job.stat.compute(reshaped)

    logging.info("Computing coefficients")
    job.coeffs = job.means - job.means[0]

    report.make_report(job)




def init_schema(infile=None):
    """Creates a new schema based on the given infile.

    Does not save it or make any changes to the state of the file
    system.

    """
    if isinstance(infile, str):
        infile = open(infile)
    header_line = infile.next().rstrip()    
    headers = header_line.split("\t")                
    is_feature_id = [i == 0 for i in range(len(headers))]
    is_sample     = [i != 0 for i in range(len(headers))]    

    print fix_newlines("""
I am assuming that the input file is tab-delimited, with a header line. I am also assuming that the first column ({0}) contains feature identifiers, and that the rest of the columns ({1}) contain expression levels. In a future release, we will be more flexible about input formats. In the meantime, if this assumption is not true, you will need to reformat your input file.

""".format(headers[0],
           ", ".join(headers[1:])))

    return Schema(
        is_feature_id=is_feature_id,
        is_sample=is_sample,
        column_names=headers)

def init_job(infile, factors, directory, force=False):

    if isinstance(infile, str):
        infile = open(infile)
    schema = init_schema(infile=infile)    

    for a in factors:
        schema.add_factor(a, object)

    if not os.path.isdir(directory):
        os.makedirs(directory)
    job = Job(directory)

    mode = 'w' if force else 'wx'
    try:
        out = open(job.schema_filename, mode)
        schema.save(out)
    except IOError as e:
        if e.errno == errno.EEXIST:
            raise UsageException("The schema file \"{}\" already exists. If you want to overwrite it, use the --force or -f argument.".format(filename))
        raise e

    logging.info("Copying {0} to {1}".format(
        infile.name,
        job.input_filename))
    shutil.copyfile(infile.name, job.input_filename)
    return job

def do_setup(args):
    job = init_job(
        infile=args.infile,
        directory=args.directory,
        factors=args.factor,
        force=args.force)

    print fix_newlines("""
I have generated a schema for your input file, with factors {factors}, and saved it to "{filename}". You should now edit that file to set the factors for each sample. The file contains instructions on how to edit it.
""").format(factors=job.schema.factors.keys(),
            filename=job.schema_filename)


def get_arguments():
    """Parse the command line options and return an argparse args
    object."""
    
    uberparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    uberparser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Be verbose (print INFO level log messages)")

    uberparser.add_argument(
        '--debug', '-d', 
        action='store_true',
        help="Print debugging information")

    subparsers = uberparser.add_subparsers()

    # Setup
    setup_parser = subparsers.add_parser(
        'setup',
        description="""Set up the job configuration. This reads the input file and outputs a YAML file that you then need to fill out in order to properly configure the job.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    setup_parser.add_argument(
        'infile',
        help="""Name of input file""",
        default=argparse.SUPPRESS,
        type=file)
    setup_parser.add_argument(
        '--factor',
        action='append',
        required=True,
        help="""A class that can be set for each sample. You can
        specify this option more than once, to use more than one
        class.""")
    setup_parser.add_argument(
        '--directory', '-D',
        default=Job.DEFAULT_DIRECTORY,
        help="The directory to store the output data")
    setup_parser.add_argument(
        '--force', '-f',
        action='store_true',
        help="""Overwrite any existing files""")

    setup_parser.set_defaults(func=do_setup)

    # Run
    run_parser = subparsers.add_parser(
        'run',
        description="""Run the job.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    run_parser.add_argument(
        '--directory', '-d',
        default=Job.DEFAULT_DIRECTORY,
        help="""The directory to store the output data.

This directory must also contain schema.yaml file and input.txt""")
    run_parser.add_argument(
        '--model', '-m',
        help="""Specify the model. Required if there is more than one class.""")
    run_parser.add_argument(
        '--stat', '-s',
        choices=['f', 't'],
        default='f',
        help="The statistic to use")
    run_parser.set_defaults(func=do_run)

    return uberparser.parse_args()


if __name__ == '__main__':
    main()
