# We need to add the root of the pade source tree to PYTHONPATH, so we
# pick up all the libraries. Note that this is not necessary if PADE
# is installed in your path anyway.
export PYTHONPATH=../../

# Path to pade executable
PADE=pade

# Path to db file created by "pade run"
DB=pade_db.h5

# Don't be verbose. Change to -v to be verbose, or -d to turn on
# debugging output.
VERBOSITY=

all : server

db : $(DB)

pade_db.h5 : 
	$(PADE) run --paired $(VERBOSITY) --condition treated --block pig sample_data_paired.txt -o $@ --schema pade_schema.yaml

server : $(DB)
	$(PADE) view $<

clean : 
	rm -f $(DB) pade.log