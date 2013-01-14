PERF_MAX_LOG_N=4

cover : 
	 nosetests --with-coverage --cover-html --cover-package page

test :
	nosetests

clean :
	rm -f *~ *.pyc *.log tests/*~ tests/*.pyc

perf :
	mkdir -p perf_report
	./perlvspython.py setup -d perf_report --max-log-n $(PERF_MAX_LOG_N)