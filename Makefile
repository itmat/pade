PERF_MAX_LOG_N=8

cover : 
	 nosetests --with-coverage --cover-html --cover-package pade

test :
	nosetests --with-doctest

clean :
	rm -f *.log tests/*~ tests/*.pyc site.tar
	cd doc; make clean
	rm -rf doc/html/generated
	rm -rf cover
	rm -f `find . -name \*~`
	rm -f `find . -name \*.pyc`


perf_setup :
	mkdir -p perf_report
	./perlvspython.py setup -d perf_report --max-log-n $(PERF_MAX_LOG_N)

perf_report/stats.% :
	./perlvspython.py run -d perf_report --max-log-n $(PERF_MAX_LOG_N) -o $@

qsub_stats :
	qsub -l h_vmem=8g -cwd -t 2 pade_perf.sh

clean_stats :
	rm -f pade_perf.sh.* perf_report/stats*

perf : perf_report/stats.*
	python ./pade/tools/perlvspython.py report $^

site :
	rm -rf doc/generated
	sphinx-apidoc pade -o doc/generated
	cd doc; make clean html

deploy_site:

	cd doc; make html
	cd doc/_build/html; tar cf ../../../site.tar *
	git checkout gh-pages
	tar xf site.tar
	git add `tar tf site.tar`
	git commit -m 'Updating site'
	git push origin gh-pages
	git checkout master

redis :
	redis-server redis.conf

worker :
	celery --app=pade worker -l info

ubuntu_setup :
	sudo apt-get update
	sudo apt-get install git python-numpy python-scipy python-matplotlib python-h5py redis-server python-setuptools python-pip
	sudo pip install virtualenv