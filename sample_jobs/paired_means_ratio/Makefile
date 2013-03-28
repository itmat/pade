# We need to add the root of the pade source tree to PYTHONPATH, so we
# pick up all the libraries. Note that this is not necessary if PADE
# is installed in your path anyway.
export PYTHONPATH=../../

# Path to pade executable
PADE=../../bin/pade

PADE=python -m cProfile -s time ../../pade/main.py 

DB=pade_db.h5

# Don't be verbose. Change to -v to be verbose, or -d to turn on
# debugging output.
VERBOSITY=

all : server

db : $(DB)

pade_db.h5 : 
	$(PADE) run --condition treated --block pig sample_data_paired.txt --stat means_ratio

server : $(DB)
	$(PADE) server --db $<

clean : 
	rm -f $(DB) pade.log