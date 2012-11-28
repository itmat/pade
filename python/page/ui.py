import matplotlib.pyplot as plt
import numpy as np

import core
import re
import argparse
from textwrap import fill
from schema import Schema

import cmd
import logging
import sys
import errno

DEFAULT_SCHEMA="page_schema.yaml"

class UsageException(Exception):
    pass

class AttributePrompt(cmd.Cmd):

    def __init__(self, schema):
        cmd.Cmd.__init__(self)
        self.schema = schema

    def do_add(self, line):
        """Add an attribute, optionally with a type. If type is supplied, it must be a valid numpy dtype, like S100, float, or int.

Examples:

  add treated bool
  add age int

"""

        tokens = line.split()
        type_ = "S100"

        if len(tokens) > 1:
            type_ = tokens[1]
            
        self.schema.add_attribute(tokens[0], type_)

    def do_remove(self, line):
        """
Remove an attribute. For example:

  remove age
"""
        schema.drop_attribute(line)

    def do_show(self, line):
        """
        Show the attributes that are currently defined."""

        print "\nAttributes are " + str(self.schema.attributes) + "\n"


    def do_done(self, line):
        """Stop editing the schema."""
        return True

    def complete_set(self, text, line, begidx, endidx):
        values = self.schema.sample_column_names()
        return [v for v in values if v.startswith(text)]

    def do_set(self, line):
        tokens = line.split()

        settings = {}
        columns  = []

        for token in tokens:
            parts = token.split('=')
            if len(parts) == 2:
                settings[parts[0]] = parts[1]
            else:
                columns.append(parts[0])

        print "I would set " + str(settings) + " for columns " + str(columns)
        for factor in settings:
            value = settings[factor]
            print "Setting {0} to {1} for samples {2}".format(
                factor, value, str(columns))
            for column in columns:
                self.schema.set_attribute(column, factor, value)

def fix_newlines(msg):
    output = ""
    for line in msg.splitlines():
        output += fill(line) + "\n"
    return output

def show_and_pause(msg):
    
    print fix_newlines(msg)
    print "\n[Enter]\n"
    sys.stdin.readline()

def show(msg):
    print fix_newlines(msg)


def do_interactive(args):
    prompt = AttributePrompt(schema)
    prompt.prompt = "-> "
    prompt.cmdloop(fix_newlines("""
I'm now going to ask you to enter all the attributes, or factors, that you want to define. Enter each attribute, one at a time, by typing "add ATTRIBUTE [TYPE]" for each attribute, specifying the type of the attribute if it is anything other than a string. You can see the attribute you've added with "show", and remove an attribute with "remove ATTRIBUTE". Type "done" when you are done adding attributes. For example, if you have three attributes, sex, age, and treated, you would enter them as follows:

-> add sex
-> add age int
-> add treated bool
-> done
"""))




def do_setup(args):
    """Ask the user for the list of factors, the values for each
    factor, and mapping from column name to factor values. Also
    establish whether the input file has a header line and a feature
    id column.
    """

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

    for a in args.attribute:
        schema.add_attribute(a, object)
    
    mode = 'w' if args.force else 'wx'

    try:
        out = open(args.schema, mode)
        schema.save(out)
    except IOError as e:
        if e.errno == errno.EEXIST:
            raise UsageException("The schema file \"{}\" already exists. If you want to overwrite it, use the --force or -f argument.".format(args.schema))
        raise e

    print fix_newlines("""
I have generated a schema for your input file, with attributes {attributes}, and saved it to "{filename}". You should now edit that file to set the attributes for each sample. The file contains instructions on how to edit it.
""").format(attributes=schema.attributes.keys(),
            filename=args.schema)


def do_run(args):

    schema = Schema.load(args.schema, args.infile)

    if len(schema.attribute_names) > 1:
        raise UsageException("I currently only support schemas with one attribute")
    
    groups = schema.sample_groups(schema.attribute_names[0])
    
    job = core.Job(args.infile, schema)

    conditions = groups.values()

    logging.info("Using these column groupings: " +
                 str(conditions))

    condition_names = [schema.attribute_names[0] + "=" + str(x)
                       for x in groups.keys()]
    alphas = core.find_default_alpha(job)
    core.do_confidences_by_cutoff(job, alphas, args.num_bins)

    #print_counts_by_confidence(breakdown, condition_names)
    #plot_counts_by_confidence(breakdown, condition_names)

    print """
Please take a look at the tables above (or at the prettier versions in {output_dir}) and select a confidence level to use. Then run "page finish --confidence CONF" to finish the job.
""".format(output_dir=args.directory)

def show_banner():
    """Print the PaGE banner.
    
    """

    print """
------------------------------------------------------------------------------

                       Welcome to PaGE {version}
                   Microarray and RNA-Seq analysis tool

------------------------------------------------------------------------------

For help, please use {script} --help or {script} -h
""".format(version=core.__version__,
           script=__file__)



def get_arguments():
    """Parse the command line options and return an argparse args
    object."""
    
    uberparser = argparse.ArgumentParser(
        description="""Finds patterns in gene expression. This will read in a tab-delimited file, where one column contains IDs of genes or other features, and other columns contain numbers representing expression levels. Running a job is a two step process. First do "page setup INFILE". This will read the input file and print out a YAML file that you need to fill out in order to indicate how columns should be grouped. Then do "page run INFILE" to actually run the analysis.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparsers = uberparser.add_subparsers()

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
        '--schema',
        help="""Location to write schema file""",
        default=DEFAULT_SCHEMA)
    setup_parser.add_argument(
        '--attribute', '-a',
        action='append',
        required=True,
        help="""An attribute that can be set for each sample. You can
        specify this option more than once, to use more than one
        attribute.""")
    setup_parser.add_argument(
        '--force', '-f',
        action='store_true',
        help="""Overwrite any existing files""")
        
    setup_parser.set_defaults(func=do_setup)
    
    parser = subparsers.add_parser('run')
    parser.set_defaults(func=do_run)
    file_locations = parser.add_argument_group("File locations")

    file_locations.add_argument(
        'infile',
        help="""Name of the input file containing the table of
        data. Must conform to the format in the README file.""",
        default=argparse.SUPPRESS,
        type=file)

    file_locations.add_argument(
        '--schema',
        help="""Path to the schema describing the input file""",
        type=file,
        default=DEFAULT_SCHEMA)

    file_locations.add_argument(
        '--directory', '-d',
        help="""Name of the directory to write results to.""",
        default='page_results')

    file_locations.add_argument(
        "--id2info",
        help="""Name of the file containing a mapping of gene id's to
        names or descriptions.""",
        default=argparse.SUPPRESS,
        type=file)

    file_locations.add_argument(
        "--id2url",  
        help="""Name of the file containing a mapping of gene id's to urls.""",
        default=argparse.SUPPRESS,
        type=file)

    file_locations.add_argument(
        "--id-filter-file", 

        help="""If you just want to run the algorithm on a subset of
        the genes in your data file, you can put the id's for those
        genes in a file, one per line, and specify that file with
        this option.""",
        default=argparse.SUPPRESS,
        type=file)

    output_options = parser.add_argument_group("Output options")

    output_options.add_argument(
            "--output-gene-confidence-list",

            help="""Set this to output a tab delimited file that maps
            every gene to its confidence of differential expression.
            For each comparison gives separate lists for up and down
            regulation. """,
            default=argparse.SUPPRESS,
            type=file)

    output_options.add_argument(
            "--output-gene-confidence-list-combined",

            help="""Set this to output a tab delimited file that maps
            every gene to its confidence of differential expression.
            For each comparison gives one list with up and down
            regulation combined. """,
            default=argparse.SUPPRESS,
            type=file)

    output_options.add_argument(
            "--output-text",
            help="""Set this to output the results also in text format. """,
            default=argparse.SUPPRESS,
            action='store_true')

    output_options.add_argument(
            "--note",
            default=argparse.SUPPRESS,
            help="""A string that will be included at the top of the
            output file.""")

    output_options.add_argument(
            "--aux-page-size", 
            type=int,
            default=500,
            help="""A whole number greater than zero.  This specifies
            the minimum number of tags there can be in one pattern
            before the results for that pattern are written to an
            auxiliary page (this keeps the main results page from
            getting too large).""")

    design = parser.add_argument_group(
        "Study design and nature of the input data")

    design.add_argument(
        "--num-bins",
        type=int,
        default=1000,        
        help="""The number of bins to use in granularizing the
        statistic over its range. This is set to a default of 1000 and
        you probably shouldn't need to change it.""")

    design.add_argument(
        "--channels",
        type=int,
        default=argparse.SUPPRESS,        
        choices=[1,2],
        help="""Is your data one or two channels?  (note: Affymetrix
        is considered one channel).""")

    design.add_argument(
        "--design", 
        default=argparse.SUPPRESS,
        choices=['d', 'r'],
        help="""For two channel data, either set this to "r" for
        "reference" design or "d" for "direct comparisons"
        (see the documentation for more  information on this
        setting). """)

    logged = design.add_mutually_exclusive_group()

    logged.add_argument(
        "--data-is-logged", 
        default=argparse.SUPPRESS,
        action='store_true',
        dest='data_is_logged',
        help="Use this option if your data has already been log transformed.")

    logged.add_argument(
        """--data-not-logged""",
        action='store_const',
        dest='data_is_logged',
        const=False,
        default=argparse.SUPPRESS,
        help="Use this option if your data has not been log transformed. ")

    paired = design.add_mutually_exclusive_group()

    paired.add_argument(
        "--paired", 
        action='store_const',
        dest='paired',
        const=True,
        default=argparse.SUPPRESS,
        help="The data is paired.")

    paired.add_argument(
        "--unpaired", 
        action='store_const',
        dest='paired',
        const=False,
        default=argparse.SUPPRESS,
        help="The data is not paired.")

    design.add_argument(
        "--missing-value", 
        default=argparse.SUPPRESS,
        help="""If you have missing values designated by a string
        (such as \"NA\"), specify  that string with this option.  You
        can either put quotes around the string  or not, it doesn't
        matter as long as the string has no spaces."""
)

    stats = parser.add_argument_group('Statistics and parameter settings')

    stats.add_argument(
        "--level-confidence",
        type=float,
        default=argparse.SUPPRESS,
        help="""A number between 0 and 1. Generate the levels with
        this confidence. See the README file for more information on
        this parameter. This can be set separately for each group
        using --level-confidence-list (see below).  NOTE: This
        parameter can be set at the end of the run after the program
        has displayed a summary breakdown of how many genes are found
        with what confidence. To do this either set the command line
        option to "L" (for "later"), or do not specify this command
        line option and enter "L" when the program prompts for the
        level confidence.""")

    stats.add_argument(
        "--level-confidence-list",
        default=argparse.SUPPRESS,
        help="""Comma-separated list of confidences. If there are more
        than two conditions (or more than one direct comparision),
        each position in the pattern can have its own confidence
        specified by this list. E.g. if there are 4 conditions, the
        list might be .8,.7,.9 (note four conditions gives patterns of
        length 3)""")

    stats.add_argument(
        "--min-presence",
        default=argparse.SUPPRESS,
        help="""A positive integer specifying the minimum number of
        values a tag should  have in each condition in order to not
        be discarded.  This can be set  separately for each condition
        using --min-presence-list """)

    stats.add_argument(
        "--min-presence-list",
        default=argparse.SUPPRESS,
        help="""Comma-separated list of positive integers, one for
        each condition,  specifying the minimum number of values a
        tag should have, for each  condition, in order not to be
        discarded.  E.g. if there are three  conditions, the list
        might be 4,6,3 """)

    use_logged = design.add_mutually_exclusive_group()

    use_logged.add_argument(
        "--use-logged-data",
        default=argparse.SUPPRESS,
        action='store_true',
        help="""Use this option to run the algorithm on the logged
        data (you can only  use this option if using the t-statistic
        as statistic).  Logging the  data usually give better
        results, but there is no rule.  Sometimes  different genes
        can be picked up either way.  It is generally best,  if using
        the t-statistic, to go with the logged data.  You might try 
        both ways and see if it makes much difference.  Both ways give
        valid  results, what can be effected is the power. """)

    use_logged.add_argument(
        "--use-unlogged-data",
        default=argparse.SUPPRESS,
        action='store_true',
        help="""Use this option to run the algorithm on the unlogged
        data.  (See  --use-loggged-data option above for more
        information.) """)

    the_stat = stats.add_mutually_exclusive_group()

    the_stat.add_argument(
        "--tstat",
        action='store_true',
        default=argparse.SUPPRESS,
        help="Use the t-statistic as statistic. ")

    the_stat.add_argument(
        "--means",
        action='store_true',
        default=argparse.SUPPRESS,
        help="Use the ratio of the means of the two groups as statistic. ")

    stats.add_argument(
        "--tstat-tuning-parameter",
        default=argparse.SUPPRESS,
        help="""The value of the t-statistic tuning
        parameter.  This is set to  a default value determined
        separately for each pattern position, but can be  set by hand
        using this command.  See the documentation for more 
        information on this parameter. """)

    stats.add_argument(
        "--shift",
        default=argparse.SUPPRESS,
        help="""A real number greater than zero.  This number will be
        added to all intensities (of the unlogged data).  See the
        documentation for more on why you might use this parameter.""")

    report = subparsers.add_parser(
        'report',
        description="""Generate an HTML report""")

    report.add_argument(
        'infile',
        help="""Name of input file""",
        default=argparse.SUPPRESS,
        type=file)
    
    report.add_argument(
        '--confidence',
        help="The confidence level.",
        required=True,
        type=float)
    report.add_argument(
        '--schema',
        help="""Location of schema file""",
        default=DEFAULT_SCHEMA)

    report.set_defaults(func=do_report)

    try:
        return uberparser.parse_args()
    except IOError as e:
        print e
        print ""
        exit(1)


def main():
    """Run PaGE."""

    logging.basicConfig(filename='page.log',
                        level=logging.INFO)
    logging.info('Page starting')

    show_banner()
    args = get_arguments()
    try:
        args.func(args)
    except UsageException as e:
        print fix_newlines("ERROR: " + e.message)
    
    logging.info('Page finishing')

def plot_counts_by_confidence(breakdown, condition_names):

    print "Plotting count of up-regulated features by confidence"
    for c in range(1, len(breakdown)):
        levels = breakdown[c, :, 0]
        up     = breakdown[c, :, 1]
        plt.plot(levels, up, ls='-', label=condition_names[c] + " up")

    print "Plotting count of down-regulated features by confidence"
    for c in range(1, len(breakdown)):
        levels = breakdown[c, :, 0]
        down   = breakdown[c, :, 2]
        plt.plot(levels, down, ls='-.', label=condition_names[c] + " down")

    plt.title("Up- and down-regulated features by confidence level")
    plt.xlabel("Confidence level")
    plt.ylabel("Num. genes")
    plt.legend(loc=1)
    plt.savefig("conf")
    plt.clf()

def write_results(directory='.'):
    pass


def print_counts_by_confidence(breakdown, condition_names):

    """Breakdown is an (n x levels x 3) table, where n is the number
    of conditions and levels is the number of confidence levels. It
    represents a list of tables, one for each condition, containing
    the confidence level, the number of up-regulated features, and the
    number of down-regulated features for each confidence level.
    """

    (n, levels, cols) = np.shape(breakdown)
    
    for c in range(1, n):
        print """
----------------------------
{:s}
{:10s} {:7s} {:7s}
----------------------------
""".format(str(condition_names[c]), 'confidence', 'num. up', 'num. down')

        for row in breakdown[c]:
            (level, up, down) = row
            print "{:10.2f} {:7d} {:9d}".format(level, int(up), int(down))


def do_report(args):
    job = core.Job(args.infile, args.schema)

    gene_conf_u = np.load("gene_conf_u.npy")
    gene_conf_u = np.load("gene_conf_u.npy")
    all_alpha   = np.load("alpha.npy")

    print "Made a job"
    print np.shape(gene_conf_u)

    # Find the value of alpha that gives the highest number of up and
    # down regulated features for this confidence level.

    (num_alphas, m, n) = np.shape(gene_conf_u)

    count_by_alpha = np.zeros((num_alphas, n))

    # We need

    # 42, 39, 38
    for c in range(n):
        for i in range(num_alphas):
            feature_stats = gene_conf_u[i,:,c]
            count_by_alpha[i, c] = len(feature_stats[feature_stats >= args.confidence])

    alpha_indices = np.argmax(count_by_alpha, axis=0)
    print alpha_indices
    print all_alpha
    print job.feature_ids

    for i, fid in enumerate(job.feature_ids):
        pass

if __name__ == '__main__':
    main()

