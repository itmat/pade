# External imports
import logging
import argparse
import os
import errno
import shutil

from textwrap import fill

# PaGE imports
from schema import Schema

class UsageException(Exception):
    pass


def fix_newlines(msg):
    output = ""
    for line in msg.splitlines():
        output += fill(line) + "\n"
    return output

class PersistedJob:

    DEFAULT_DIRECTORY = "pageseq_out"

    def __init__(self, directory):
        self.directory = directory
        
    @property
    def schema_filename(self):
        return os.path.join(self.directory, 'schema.yaml')

    @property
    def input_filename(self):
        return os.path.join(self.directory, 'input.txt')
    
    @property
    def schema(self):
        return Schema.load(open(self.schema_filename), self.input_filename)


def main():
    """Run pageseq."""

    logging.basicConfig(filename='page.log',
                        level=logging.INFO)
    logging.info('Page starting')

    args = get_arguments()
    try:
        args.func(args)
    except UsageException as e:
        print fix_newlines("ERROR: " + e.message)
    
    logging.info('Page finishing')    

def do_run(args):

    job    = PersistedJob(args.directory)
    schema = job.schema
    print "Factors are {0}".format(schema.factors)

def do_setup(args):
    header_line = args.infile.next().rstrip()
    headers = header_line.split("\t")

    is_feature_id = [i == 0 for i in range(len(headers))]
    is_sample     = [i != 0 for i in range(len(headers))]

    print fix_newlines("""
I am assuming that the input file is tab-delimited, with a header line. I am also assuming that the first column ({0}) contains feature identifiers, and that the rest of the columns ({1}) contain expression levels. In a future release, we will be more flexible about input formats. In the meantime, if this assumption is not true, you will need to reformat your input file.

""".format(headers[0],
           ", ".join(headers[1:])))

    schema = Schema(
        is_feature_id=is_feature_id,
        is_sample=is_sample,
        column_names=headers)

    for a in args.factor:
        schema.add_factor(a, object)
    
    mode = 'w' if args.force else 'wx'

    if not os.path.isdir(args.directory):
        os.makedirs(args.directory)
    job = PersistedJob(args.directory)

    try:
        out = open(job.schema_filename, mode)
        schema.save(out)
    except IOError as e:
        if e.errno == errno.EEXIST:
            raise UsageException("The schema file \"{}\" already exists. If you want to overwrite it, use the --force or -f argument.".format(filename))
        raise e

    print "Copying {0} to {1}".format(
        args.infile.name,
        job.input_filename)
    shutil.copyfile(args.infile.name, job.input_filename)

    print fix_newlines("""
I have generated a schema for your input file, with factors {factors}, and saved it to "{filename}". You should now edit that file to set the factors for each sample. The file contains instructions on how to edit it.
""").format(factors=schema.factors.keys(),
            filename=job.schema_filename)

def get_arguments():
    """Parse the command line options and return an argparse args
    object."""
    
    uberparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

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
        '--directory', '-d',
        default=PersistedJob.DEFAULT_DIRECTORY,
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
        default=PersistedJob.DEFAULT_DIRECTORY,
        help="""The directory to store the output data.

This directory must also contain schema.yaml file and input.txt""")
    run_parser.add_argument(
        '--model', '-m',
        help="""Specify the model. Required if there is more than one class.""")
    run_parser.set_defaults(func=do_run)
        
        

    return uberparser.parse_args()


if __name__ == '__main__':
    main()
