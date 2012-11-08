import numpy as np
from textwrap import fill
from io import StringIO
from numpy.lib.recfunctions import append_fields
import yaml

class Schema(object):

    def __init__(self, 
                 attributes=[],
                 is_feature_id=None,
                 is_sample=None,
                 column_names=None):
        """Construct a Schema. 

  attributes    - a list of allowable attributes for the sample.

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

        self.attributes = {}
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

        for (name, dtype) in attributes:
            self.add_attribute(name, dtype)

    @property
    def attribute_names(self):
        """Return a list of the attribute names for this schema."""

        return [x for x in self.attributes]

    def sample_column_names(self):
        """Return a list of the names of columns that contain
        intensities."""

        return self.column_names[self.is_sample]

    def add_attribute(self, name, dtype):
        """Add an attribute with the given name and data type, which
        must be a valid numpy dtype."""

        default = None
        if dtype == "int":
            default = 0
        else:
            default = ""
        values = [default for sample in self.sample_column_names()]

        if self.table is None:
            self.table = [(value,) for value in values]
            self.table = np.array(self.table, dtype=[(name, dtype)])
        else:
            self.table=(append_fields(self.table, name, values, dtype))

        self.attributes[name] = dtype
        
    def drop_attribute(self, name):
        """Remove the attribute with the given name."""

        self.table = drop_fields(name)
        del self.attributes[name]

    @classmethod
    def load(cls, stream):
        """Load a schema from the specified stream, which must
        represent a YAML document in the format produced by
        Schema.dump. The type of stream can be any type accepted by
        yaml.load."""

        doc = yaml.load(stream)

        # Build the arrayrs of column names, feature id booleans, and
        # sample booleans
        cols = doc['columns']
        col_names = [col['name'] for col in cols]
        col_types = np.array([col['type'] for col in cols])
        is_feature_id = col_types == 'feature_id'
        is_sample     = col_types == 'sample'

        schema = Schema(
            column_names=col_names,
            is_feature_id=is_feature_id,
            is_sample=is_sample)

        # Now add all the attributes and their types
        attributes = doc['attributes']
        for attribute in attributes:
            schema.add_attribute(attribute['name'], 
                                 attribute['dtype'])

        for sample, attrs in doc['sample_attribute_mapping'].iteritems():
            for name, value in attrs.iteritems():
                schema.set_attribute(sample, name, value)

        return schema
    
    def save(self, out):
        """Save the schema to the specified file."""

        # Need to convert column names to strings, from whatever numpy
        # type they're stored as.
        names = [str(name) for name in self.column_names]

        sample_cols = {}
        columns = []

        for i, name in enumerate(names):

            col = { "name" : name }

            if self.is_feature_id[i]:
                col["type"] = "feature_id"

            elif self.is_sample[i]:
                col["type"] = "sample"
                
                sample_cols[name] = {}
                for attribute in self.attributes:
                    value = self.get_attribute(name, attribute)
                    if self.attributes[attribute].startswith("S"):
                        value = str(value)

                    if type(value) == str:
                        pass
                    elif type(value) == np.int32:
                        value = int(value)
                    elif type(value) == np.bool_:
                        value = bool(value)
                    
                    sample_cols[name][attribute] = value

            columns.append(col)

        attributes = [ { "name"  : name, "dtype" : type_ } 
                       for (name, type_) in self.attributes.iteritems()]

        doc = {
            "attributes"               : attributes,
            "columns"                  : columns,
            "sample_attribute_mapping" : sample_cols,
            }

        data = yaml.dump(doc, default_flow_style=False, encoding=None)

        for line in data.splitlines():
            if (line == "attributes:"):
                write_yaml_block_comment(out, """This lists all the attributes defined for this file.
""")

            elif (line == "columns:"):
                out.write(unicode("\n"))
                write_yaml_block_comment(out, """This lists all of the columns present in the input file, each with its name and type. name is taken directly from the input file's header line. Type must be either "feature_id", "sample", or null.
""")

            elif (line == "sample_attribute_mapping:"):
                out.write(unicode("\n"))
                write_yaml_block_comment(out, """This maps each column name (for columns that represent samples) to a mapping from attribute name to value.""")

            out.write(unicode(line) + "\n")


    def set_attribute(self, sample_name, attribute, value):
        """Set an attribute for a sample, identified by sample
        name."""

        sample_num = self.sample_num(sample_name)
        self.table[sample_num][attribute] = value

    def get_attribute(self, sample_name, attribute):
        """Get an attribute for a sample, identified by sample
        name."""

        sample_num = self.sample_num(sample_name)
        value = self.table[sample_num][attribute]
        if self.attributes[attribute].startswith("S"):
            value = str(value)
        return value

    def sample_num(self, sample_name):
        """Return the sample number for sample with the given
        name. The sample number is the index into the table for the
        sample."""

        return self.sample_name_index[sample_name]

    def sample_groups(self, attribute=None):
        """Returns a dictionary mapping each value of attribute to the
        list of sample numbers that have that value set."""

        grouping = {}
        for i, val in enumerate(self.table[attribute]):
            if val not in grouping:
                grouping[val] = []
            grouping[val].append(i)

        return grouping


def write_yaml_block_comment(fh, comment):
    fh.write(unicode(fill(comment,
                  initial_indent="# ",
                  subsequent_indent="# ")))
    fh.write(unicode("\n"))

