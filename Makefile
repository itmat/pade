PERF_MAX_LOG_N=8

cover : 
	 nosetests --with-coverage --cover-html --cover-package page

test :
	nosetests --with-doctest

clean :
	rm -f *~ **/*~ **/*.pyc *.pyc *.log tests/*~ tests/*.pyc 
	cd doc; make clean
	rm -rf doc/html/generated
	rm -rf cover

perf_setup :
	mkdir -p perf_report
	./perlvspython.py setup -d perf_report --max-log-n $(PERF_MAX_LOG_N)

perf_report/stats.% :
	./perlvspython.py run -d perf_report --max-log-n $(PERF_MAX_LOG_N) -o $@

qsub_stats :
	qsub -l h_vmem=8g -cwd -t 2 page_perf.sh

clean_stats :
	rm -f page_perf.sh.* perf_report/stats*

perf : perf_report/stats.*
	python ./page/tools/perlvspython.py report $^

site :
	cd doc; make html

deploy_site:

	cd doc; make html
	cd doc/_build/html; tar cf ../../../../site.tar *
	git checkout gh-pages
	tar xf site.tar
	git add `tar tf site.tar`
	git commit -m 'Updating site'
	git push origin gh-pages
	git checkout master

