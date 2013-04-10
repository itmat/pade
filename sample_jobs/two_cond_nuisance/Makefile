# We need to add the root of the pade source tree to PYTHONPATH, so we
# pick up all the libraries. Note that this is not necessary if PADE
# is installed in your path anyway.
export PYTHONPATH=../../

# Path to pade executable
PADE=../../bin/pade

# Path to db file created by "pade run"
DB=pade_db.h5

# Don't be verbose. Change to -v to be verbose, or -d to turn on
# debugging output.
VERBOSITY=

all : server

default.h5 : 
	$(PADE) run $(VERBOSITY) --condition treated --block sex sample_data_2_cond_nuisance.txt -o $@ --schema pade_schema.yaml

glm_gaussian.h5 : 
	$(PADE) run $(VERBOSITY) --condition treated --block sex sample_data_2_cond_nuisance.txt -o $@ --schema pade_schema.yaml --stat glm --glm-family gaussian

server : $(DB)
	$(PADE) server --db $<

clean : 
	rm -f $(DB) pade.log