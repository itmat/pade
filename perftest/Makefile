PYTHON=PYTHONPATH=.. python
PAGE=$(PYTHON) ../bin/page

feature_counts=\
   000000177 000000316 000000056 0000001000 \
   000001778 000003162 000005623 0000010000 \
   000017782 000031622 000056234 0000100000 \
   000177827 000316227 000562341 0001000000
#  001778279 003162277 005623413 0010000000
#  017782794 031622776 056234132 0100000000
#  177827941 316227766 562341325 1000000000

INPUTS=$(foreach n,$(feature_counts),in_$(n).txt)
TABLES=$(foreach n,$(feature_counts),job_$(n)/data/table.npy)
REPORTS=$(foreach n,$(feature_counts),job_$(n)/html/index.html)

input : $(INPUTS)
setup : $(TABLES)
run   : $(REPORTS)

in_%.txt : ../sample_data/sample_data_4_class.txt
	PYTHONPATH=.. python ../page/tools/make_perf_input.py -n $* $< >$@

job_%/data/table.npy : in_%.txt
	$(PAGE) setup --factor level $< --directory job_$*
	cp schema.yaml job_$*/schema.yaml

job_%/html/index.html : job_%/data/table.npy
	$(PAGE) run --directory job_$* --full-model level

clean :
	rm -rf in_*.txt job_*