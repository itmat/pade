PERF_MAX_LOG_N=8

cover : 
	 nosetests --with-coverage --cover-html --cover-package page

test :
	nosetests

clean :
	rm -f *~ *.pyc *.log tests/*~ tests/*.pyc

perf_setup :
	mkdir -p perf_report
	./perlvspython.py setup -d perf_report --max-log-n $(PERF_MAX_LOG_N)

perf_report/stats.% :
	./perlvspython.py run -d perf_report --max-log-n $(PERF_MAX_LOG_N) -o $@

qsub_stats :
	qsub -l h_vmem=8g -cwd -t 2 page_perf.sh

clean_stats :
	rm -f page_perf.sh.* perf_report/stats*

perf : perf_report/stats
	./perlvspython.py report $^