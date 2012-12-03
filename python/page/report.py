

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
from lxml.html import builder as E
import lxml
import os
from jinja2 import Environment, PackageLoader

def ensure_increases(a):
    """Given an array, return a copy of it that is monotonically
    increasing."""

    for i in range(len(a) - 1):
        a[i+1] = max(a[i], a[i+1])

def ensure_decreases(a):
    for i in range(len(a) - 1):
        pass



class Report:
    def __init__(self, job, output_dir, results):
        self.job           = job
        self.results       = results
        self.output_dir    = output_dir
        self.stats         = results.stats
        self.conf_levels   = results.conf_levels

        self.unperm_counts = results.up.unperm_counts
        self.raw_conf      = results.up.raw_conf
        self.conf_to_count = results.up.conf_to_count
        self.best_params   = results.up.best_params

    def make_report(self):
        cwd = os.getcwd()
        print "Condition names are " + str(self.job.condition_names)
        try:
            os.chdir(self.output_dir)
            self.make_jinja_report()
        finally:
            os.chdir(cwd)


    def make_jinja_report(self):
        env = Environment(loader=PackageLoader('page'))

        stats = self.stats
        output_dir = self.output_dir

        stat_hists = []
        (s, C, n) = np.shape(stats)
    
        maxstat = np.max(stats)
        minstat = np.min(stats)

        raw_conf_plots = []

        print "Plotting histograms"
        for i in range(s):
            for c in range(1, C):
                filename = 'stat_hist_test_{test}_class_{cls}.png'.format(
                    test=i, cls=c)
                plt.cla()
                plt.hist(stats[i, c], bins=50)
                plt.xlim([minstat, maxstat])
                plt.xlabel('Statistic')
                plt.xlabel('Features')
                plt.title('Number of features by statistic, test {test}, class {cls}'.format(test=i, cls=self.job.condition_names[c]))
                plt.savefig(filename) 
                stat_hists.append(filename)

        print "Making stat hist pages"
        classes = []
        for c in range(1, C):
            with open('stat_hist_class_{cls}.html'.format(cls=c), 'w') as out:
                template = env.get_template('stat_hist_class.html')
                stat_hists = [
                    'stat_hist_test_{test}_class_{cls}.png'.format(
                        test=test, cls=c)
                    for test in range(s)]
                out.write(template.render(
                        job=self.job,
                        stat_hists=stat_hists,
                        condition_num=c,
                        ))

        print "Making index"
        with open('index.html', 'w') as out:
            template = env.get_template('index.html')
        
            out.write(template.render(
                    condition_nums=range(1, C),
                    results=self.results,
                    job=self.job))


    def make_report_in_dir(self):        
        stats = self.stats
        output_dir = self.output_dir

        stat_hists = []
        (s, m, n) = np.shape(stats)
    
        maxstat = np.max(stats)
        minstat = np.min(stats)

        raw_conf_plots = []

        for c in range(1, n):
            plt.cla()
            filename = 'raw_conf_class_{cls}.png'.format(cls=c)

            param_idxs = self.best_params[:, c]
            
            idxs = [(param_idxs[i], i, c) for i in range(len(self.conf_levels))]

#            best_counts = self.unperm_counts[idxs]

            x = [self.conf_to_count[i] for i in idxs]
            y = self.conf_levels
            
            plt.plot(x, y)

            print "Shape of unperm counts is " + str(np.shape(self.unperm_counts))
            print "Best counts are " + str(idxs)
            print "X is " + str(x)
            print "Y is " + str(y)
            for i in range(s):
                x = self.unperm_counts[i, :, c]
                y = self.raw_conf[i, :, c] 
#                plt.plot(x, y,
#                         label='Test {test}'.format(test=i))

                x = self.conf_to_count[i, c]
                y = self.conf_levels

                plt.plot(x, y,
                         linestyle='dashed',
                         label='Test {test} summary'.format(test=i))

            plt.xlabel('Features up-regulnated')
            plt.ylabel('Confidence (raw, may not be monotonically increasing)')
            plt.title("Raw confidence level by up-regulated features\nclass {cls}".format(cls=c))
            plt.ylim([np.min(self.conf_levels), np.max(self.conf_levels)])
            plt.xlim([0,
                      np.max(self.conf_to_count)])
            plt.savefig(filename)
            raw_conf_plots.append(E.IMG(src=filename, width='400'))
        
        plots = []
        plots.extend(stat_hists)
        plots.extend(raw_conf_plots)
            
        html = E.HTML(
            E.HEAD(
                E.LINK(rel='stylesheet', href="page.css", type="text/css"),
                E.TITLE("PaGE Output")),
            E.BODY(
                E.H2('Distribution of statistic values for features'),
                *plots))

        with open('index.html', 'w') as out:
            out.write(lxml.html.tostring(html))
