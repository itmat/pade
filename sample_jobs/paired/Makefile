PADE=PYTHONPATH=../../ ../../bin/pade
DB=pade_db.h5

all : server

db : $(DB)

pade_db.h5 : 
	$(PADE) run --paired -d --full-model 'treated * pig' --reduced-model 'pig' sample_data_2_class.txt --db $(DB)

server : $(DB)
	$(PADE) server --db $<

clean : 
	rm -f $(DB)