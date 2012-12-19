import numpy as np
from collections import OrderedDict
from textwrap import fill
from io import StringIO
from numpy.lib.recfunctions import append_fields
import yaml

class Schema(object):

    def __init__(self, 
                 factors=[],
                 is_feature_id=None,
                 is_sample=None,
                 column_names=None):
        """Construct a Schema. 

  factors    - a list of allowable factors for the sample.

  column_names  - a list of strings, giving names for the columns.

  is_feature_id - a list of booleans of the same length as
                  column_names. is_feature_id[i] indicates if the ith
                  column contains feature ids (e.g. gene names).

  is_sample     - a list of booleans of the same length as
                  column_names. is_sample[i] indicates if the ith
                  column contains a sample.

  filename      - the filename for the schema.

Any columns for which is_feature_id is true will be treated as feature
ids, and any for which is_sample is true will be assumed to contain
intensity values. No column should have both is_feature_id and
is_sample set to true. Any columns where both is_feature_id and
is_sample are false will simply be ignored.

  """

        if column_names is None:
            raise Exception("I need column names")
        else:
            column_names = np.array(column_names)

        self.factors = OrderedDict()
        self.is_feature_id = np.array(is_feature_id, dtype=bool)
        self.is_sample     = np.array(is_sample,     dtype=bool)
        self.column_names  = column_names
        self.table = None
        self.sample_to_column  = []
        self.sample_name_index = {}

        for i in range(len(self.is_sample)):
            if self.is_sample[i]:
                n = len(self.sample_to_column)
                self.sample_name_index[column_names[i]] = n
                self.sample_to_column.append(i)

        for (name, dtype) in factors:
            self.add_factor(name, dtype)

    @property
    def factor_names(self):
        """Return a list of the factor names for this schema."""

        return [x for x in self.factors]

    def sample_column_names(self):
        """Return a list of the names of columns that contain
        intensities."""

        return self.column_names[self.is_sample]

    def add_factor(self, name, dtype):
        """Add an factor with the given name and data type, which
        must be a valid numpy dtype."""
        
        default = None
        if dtype == "int":
            default = 0
        else:
            default = None

        new_table = []

        self.factors[name] = dtype

        if self.table is None:
            new_table = [(default,) for s in self.sample_column_names()]

        else:
            for row in self.table:
                row = tuple(row) + (default,)
                new_table.append(row)
            
        self.table = np.array(new_table, dtype=[(k, v) for k, v in self.factors.iteritems()])
        
    def drop_factor(self, name):
        """Remove the factor with the given name."""

        self.table = drop_fields(name)
        del self.factors[name]

    @classmethod
    def load(cls, stream, infile):
        """Load a schema from the specified stream, which must
        represent a YAML document in the format produced by
        Schema.dump. The type of stream can be any type accepted by
        yaml.load."""
        col_names = None

        if isinstance(infile, str):
            infile = open(infile)

        header_line = infile.next().rstrip()
        col_names = header_line.split("\t")

        doc = yaml.load(stream)

        # Build the arrays of column names, feature id booleans, and
        # sample booleans
        feature_id_cols = set(doc['feature_id_columns'])

        is_feature_id = [c in feature_id_cols for c in col_names]
        is_sample     = [c in doc['sample_factor_mapping'] for c in col_names]

        schema = Schema(
            column_names=col_names,
            is_feature_id=is_feature_id,
            is_sample=is_sample)

        # Now add all the factors and their types
        factors = doc['factors']
        for factor in factors:
            dtype = factor['dtype'] if 'dtype' in factor else object
            schema.add_factor(factor['name'], 
                                 dtype)

        for sample, attrs in doc['sample_factor_mapping'].iteritems():
            for name, value in attrs.iteritems():
                schema.set_factor(sample, name, value)

        return schema
    
    def save(self, out):
        """Save the schema to the specified file."""

        # Need to convert column names to strings, from whatever numpy
        # type they're stored as.
        names = [str(name) for name in self.column_names]

        sample_cols     = {}
        feature_id_cols = []

        for i, name in enumerate(names):

            if self.is_feature_id[i]:
                feature_id_cols.append(name)

            elif self.is_sample[i]:
                
                sample_cols[name] = {}
                for factor in self.factors:
                    value = self.get_factor(name, factor)
#                    if self.factors[factor].startswith("S"):
#                        value = str(value)

                    if type(value) == str:
                        pass
                    elif type(value) == np.int32:
                        value = int(value)
                    elif type(value) == np.bool_:
                        value = bool(value)
                    
                    sample_cols[name][factor] = value

        factors = []
        for name, type_ in self.factors.iteritems():
            a = { "name" : name }
            if type_ != object:
                a['dtype'] = type_
            factors.append(a)

        doc = {
            "factors"               : factors,
            "feature_id_columns"       : feature_id_cols,
            "sample_factor_mapping" : sample_cols,
            }

        data = yaml.dump(doc, default_flow_style=False, encoding=None)

        for line in data.splitlines():
            if (line == "factors:"):
                write_yaml_block_comment(out, """This lists all the factors defined for this file.
""")

            elif (line == "feature_id_columns:"):
                out.write(unicode("\n"))
                write_yaml_block_comment(out, """This lists all of the columns that contain feature IDs (for example gene ids).""")

            elif (line == "sample_factor_mapping:"):
                out.write(unicode("\n"))
                write_yaml_block_comment(out, """This sets all of the factors for all columns that represent samples. You should fill out this section to set each factor for each sample. For example, if you had a sample called sample1 that recieved some kind of treatment, and one called sample2 that did not, you might have:

sample_factor_mapping:
  sample1:
    treated: yes
  sample2:
    treated: no

""")

            out.write(unicode(line) + "\n")


    def set_factor(self, sample_name, factor, value):
        """Set an factor for a sample, identified by sample
        name."""

        sample_num = self.sample_num(sample_name)
        self.table[sample_num][factor] = value

    def get_factor(self, sample_name, factor):
        """Get an factor for a sample, identified by sample
        name."""

        sample_num = self.sample_num(sample_name)
        value = self.table[sample_num][factor]
#        if self.factors[factor].startswith("S"):
#            value = str(value)
        return value

    def sample_num(self, sample_name):
        """Return the sample number for sample with the given
        name. The sample number is the index into the table for the
        sample."""

        return self.sample_name_index[sample_name]

    def sample_groups(self, factor):
        """Returns a dictionary mapping each value of factor to the
        list of sample numbers that have that value set."""

        grouping = OrderedDict()
        
        for i, val in enumerate(self.table[factor]):
            key = (factor, val)
            if key not in grouping:
                grouping[key] = []
            grouping[key].append(i)

        return grouping

    def condition_name(self, c):
        """Return a name for condition c, based on the factor values for that condition"""
        pass
        

def write_yaml_block_comment(fh, comment):
    result = ""
    for line in comment.splitlines():
        result += fill(line, initial_indent = "# ", subsequent_indent="# ")
        result += "\n"
    fh.write(unicode(result))

