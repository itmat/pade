import unittest
from pade.metadb import JobMeta, MetaDB
from redis import Redis
from pade.http.server import PadeViewer, PadeRunner
import pade.http.jobdetails
import pade.config
import tempfile
import shutil
import time
import os
import redis.exceptions
import logging

standard_routes = [
    '/',
    '/jobs',
    '/jobs/0/',
    '/jobs/0/conf_level/9',
    '/jobs/0/features/14',
    '/jobs/0/features/14/measurement_scatter',
    '/jobs/0/features/14/measurement_bars',
    '/jobs/0/mean_vs_std',
    '/jobs/0/stat_dist',
    '/jobs/0/stat_dist/1.png',
    '/jobs/0/confidence_dist',
    '/jobs/0/feature_count_and_score_by_stat.html',
    '/jobs/0/bin_to_score.png',
    '/jobs/0/bin_to_features.png',
    '/jobs/0/conf_dist',
    '/jobs/0/score_dist_for_tuning_params.png',
    ]

class PadeViewerTestCase(unittest.TestCase):
    
    def setUp(self):
        self.app = PadeViewer().test_client()

        pade.http.jobdetails.job_dbs = [ 
            JobMeta(0, None, path, imported=True)
            for path in [ 'sample_jobs/test/server_test_job.pade' ] ]

    def assertOk(self, route):
        self.assertStatus(route, 200)

    def assertStatus(self, route, status):
        rv = self.app.get(route)
        self.assertEquals(rv.status_code, status)

    def test_routes(self):

        for route in standard_routes:
            self.assertOk(route)

        for route in ['/jobs/0/features/14/interaction_plot']:
            self.assertStatus(route, 404)

class PadeRunnerTestCase(unittest.TestCase):
    
    def setUp(self):
        (this_dir, this_file) = os.path.split(__file__)
        config = pade.config.test

        try:
            redis_db = Redis(host=config.metadb_host,
                             port=config.metadb_port,
                             db=config.metadb_db)
            redis_db.flushdb()

        except redis.exceptions.ConnectionError as e:
            self.skipTest("No redis")
            return

        self.mdb = MetaDB(config.metadb_dir, redis_db)
        self.mdb.redis.flushdb()
        self.app = PadeRunner(config).test_client()

    def test_setup_job(self):
        self.assertOk("/")


    def assertOk(self, route):
        self.assertStatus(route, 200)

    def assertStatus(self, route, status):
        rv = self.app.get(route)
        self.assertEquals(rv.status_code, status)

    def test_new_job_workflow(self):
        r = self.app.get("/new_job/clear_workflow")
        self.assertEquals(r.status_code, 302)

        r = self.app.get("/input_files/")
        self.assertEquals(r.status_code, 200)

        r = self.app.get("/input_files/upload_raw_file")
        self.assertEquals(r.status_code, 200)

        with open('sample_jobs/two_cond_nuisance/sample_data_2_cond_nuisance.txt') as f:
            infile = self.mdb.add_input_file(name="test.txt", description="Some comments", stream=f)

        r = self.app.get("/input_files/" + str(infile.obj_id))
        self.assertEquals(r.status_code, 200)
        self.assertTrue('c1r1' in r.data)

        with open(infile.path) as f:
            self.assertTrue('c1r2' in f.next())
            
        r = self.app.get("/new_job/select_input_file?input_file_id=" + str(infile.obj_id),
                         follow_redirects=True)
        self.assertTrue('Feature ID' in r.data)
        self.assertTrue('Sample' in r.data)

        r = self.app.post("/new_job/column_roles",
                          data={
                'roles-0': 'feature_id',
                'roles-1': 'sample',
                'roles-2': 'sample',
                'roles-3': 'sample',
                'roles-4': 'sample',
                'roles-5': 'sample',
                'roles-6': 'sample',
                'roles-7': 'sample',
                'roles-8': 'sample',
                'roles-9': 'sample',
                'roles-10': 'sample',
                'roles-11': 'sample',
                'roles-12': 'sample',
                'roles-12': 'sample',
                'roles-13': 'sample',
                'roles-14': 'sample',
                'roles-15': 'sample',
                'roles-16': 'sample',
                },
                          follow_redirects=True)
        self.assertEquals(r.status_code, 200)
        self.assertTrue('Factor name' in r.data)
        self.assertTrue('Possible values' in r.data)

        r = self.app.post("/new_job/add_factor",
                          data={
                'factor_name': 'treated',
                'possible_values-0': 'no',
                'possible_values-1': 'yes',
                'possible_values-2': '',
                'possible_values-3': '',},
                          follow_redirects=True)

        self.assertEquals(r.status_code, 200)
        self.assertTrue('treated' in r.data)
        self.assertTrue('no' in r.data)
        self.assertTrue('yes' in r.data)

        r = self.app.post("/new_job/add_factor",
                          data={
                'factor_name': 'gender',
                'possible_values-0': 'male',
                'possible_values-1': 'female',
                'possible_values-2': '',
                'possible_values-3': '',},
                          follow_redirects=True)

        self.assertEquals(r.status_code, 200)
        self.assertTrue('treated' in r.data)
        self.assertTrue('no' in r.data)
        self.assertTrue('yes' in r.data)
        self.assertTrue('male' in r.data)
        self.assertTrue('female' in r.data)

        r = self.app.post("/new_job/column_labels",
                          data={
                'assignments-0' : 'no',
                'assignments-1' : 'male',
                'assignments-2' : 'no',
                'assignments-3' : 'male',
                'assignments-4' : 'no',
                'assignments-5' : 'male',
                'assignments-6' : 'no',
                'assignments-7' : 'male',

                'assignments-8' : 'no',
                'assignments-9' : 'female',
                'assignments-10' : 'no',
                'assignments-11' : 'female',
                'assignments-12' : 'no',
                'assignments-13' : 'female',
                'assignments-14' : 'no',
                'assignments-15' : 'female',

                'assignments-16' : 'yes',
                'assignments-17' : 'male',
                'assignments-18' : 'yes',
                'assignments-19' : 'male',
                'assignments-20' : 'yes',
                'assignments-21' : 'male',
                'assignments-22' : 'yes',
                'assignments-23' : 'male',

                'assignments-24' : 'yes',
                'assignments-25' : 'female',
                'assignments-26' : 'yes',
                'assignments-27' : 'female',
                'assignments-28' : 'yes',
                'assignments-29' : 'female',
                'assignments-30' : 'yes',
                'assignments-31' : 'female'},
                          follow_redirects=True)

        self.assertEquals(r.status_code, 200)
        self.assertTrue('block and condition factors' in r.data)

        r = self.app.post("/new_job/setup_job_factors",
                          data={
                'factor_roles-0' : 'condition',
                'factor_roles-1' : 'block'},
                          follow_redirects=True)
        self.assertEquals(r.status_code, 200)
        self.assertTrue('Other settings' in r.data)

        r = self.app.post("/new_job/other_settings",
                          
                          data={
                'statistic' : 'f',
                'tuning_params': "0.001 0.01 0.1 1 3 10 30 100 300 1000 3000",
                'bins' : '1000',
                'permutations' : '1000',
                'summary_min_conf_level' : '0.1',
                'summary_step_size' : '0.05' },
                          follow_redirects=True)

        self.assertEquals(r.status_code, 200)
        self.assertTrue('confirm' in r.data)

        r = self.app.get("/new_job/submit", follow_redirects=True)
        self.assertEquals(r.status_code, 200)

        start_time = time.time()
        while True:
            time.sleep(1)

            r = self.app.get("/jobs/1/")
            if 'Features by confidence level' in r.data:
                break
            elif time.time() - start_time > 60:
                self.fail("Job is taking too long")

        for route in ['/jobs/1/features/14/interaction_plot']:
            self.assertOk(route)





if __name__ == '__main__':
    unittest.main()
