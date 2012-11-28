

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
from lxml.html import builder as E
import lxml
import os

class Report:
    def __init__(self, output_dir, stats, unperm_counts, raw_conf):
        self.output_dir = output_dir
        self.stats      = stats
        self.unperm_counts = unperm_counts
        self.raw_conf   = raw_conf

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
            for i in range(s):
                x = self.raw_conf[i, :, c] 
                y = self.unperm_counts[i, :, c]
                plt.plot(x[x >= 0], y[x >= 0], 
                         label='Test {test}'.format(test=i))
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
