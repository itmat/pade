

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
from lxml.html import builder as E
import lxml
import os


def ensure_increases(a):
    """Given an array, return a copy of it that is monotonically
    increasing."""

    for i in range(len(a) - 1):
        a[i+1] = max(a[i], a[i+1])

def ensure_decreases(a):
    for i in range(len(a) - 1):
        pass

class Report:
    def __init__(self, output_dir, stats, unperm_counts, raw_conf, conf_levels, conf_to_count, best_params):
        self.output_dir    = output_dir
        self.stats         = stats
        self.unperm_counts = unperm_counts
        self.raw_conf      = raw_conf
        self.conf_to_count  = conf_to_count
        self.conf_levels   = conf_levels
        self.best_params  = best_params

    def make_report(self):
        cwd = os.getcwd()
        try:
            os.chdir(self.output_dir)
            self.make_report_in_dir()
        finally:
            os.chdir(cwd)

    def make_report_in_dir(self):        
        stats = self.stats
        output_dir = self.output_dir

        stat_hists = []
        (s, m, n) = np.shape(stats)
    
        maxstat = np.max(stats)
        minstat = np.min(stats)

        raw_conf_plots = []

        for i in range(s):
            for c in range(1, n):
                filename = 'stat_hist_test_{test}_class_{cls}.png'.format(
                    test=i, cls=c)
                plt.cla()
                plt.hist(stats[i, :, c], bins=50)
                plt.xlim([minstat, maxstat])
                plt.xlabel('Statistic')
                plt.xlabel('Features')
                plt.title('Number of features by statistic, test {test}, class {cls}'.format(test=i, cls=c))
                plt.savefig(filename) 
                stat_hists.append(E.IMG(src=filename, width='400'))


        for c in range(1, n):
            plt.cla()
            filename = 'raw_conf_class_{cls}.png'.format(cls=c)

            best_params = self.best_params[:, c]

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

                x = self.conf_to_count[i, :, c]
                y = self.conf_levels

                plt.plot(x, y,
                         linestyle='dashed',
                         label='Test {test} summary'.format(test=i))

            plt.xlabel('Features up-regulated')
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
