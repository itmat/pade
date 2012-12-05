

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
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

        self.plot_histograms = True

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
        (s, C, N) = np.shape(stats)
    
        maxstat = np.max(stats)
        minstat = np.min(stats)

        raw_conf_plots = []

        if self.plot_histograms:
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

        colors = matplotlib.rcParams['axes.color_cycle']
        plt.clf()
        for c in range(1, C):
            plt.plot(self.results.conf_levels,
                     self.results.up.best_counts[c],
                     colors[c] + '-^',
                     label=self.job.condition_names[c] + ' up')
            plt.plot(self.results.conf_levels,
                     self.results.down.best_counts[c],
                     colors[c] + '-v',
                     label=self.job.condition_names[c] + ' down')
        plt.legend()
        plt.title('Differentially expressed features by confidence level')
        plt.xlabel('Confidence')
        plt.ylabel('Differentially expressed features')
        plt.savefig('count_by_conf')

        print "Making index"
        with open('index.html', 'w') as out:
            template = env.get_template('index.html')
        
            out.write(template.render(
                    condition_nums=range(1, C),
                    results=self.results,
                    job=self.job))

        print "Alphas is " + str(self.results.alphas)

        level = 4

        up_params   = self.results.up.best_params[:, level]
        down_params = self.results.down.best_params[:, level]

        up_stats   = self.results.best_stats_by_level('up')[level]
        down_stats = self.results.stats[(down_params, np.arange(C))]
        up_cutoffs = self.results.up_cutoffs_by_level
        print "Up cutoffs is " + str(up_cutoffs)
        is_up = np.zeros(np.shape(up_stats), bool)
        any_up = np.zeros(np.shape(up_stats)[1], bool)

        for i in range(N):
            is_up[:, i] = up_stats[:, i] >= up_cutoffs[level]
            any_up[i] = np.any(up_stats[1:, i] >= up_cutoffs[level, 1:])

        print "Conf:"
        print self.results.up.raw_conf[1, 1]

        self.results.best_stats_by_level('up')

        with open('features_by_confidence.html', 'w') as out:
            template = env.get_template('features_by_confidence.html')
            out.write(
                template.render(
                    condition_nums=range(1, C),
                    # TODO: Get this from the user
                    up_cutoffs=up_cutoffs,
                    level=4,
                    feature_nums=range(len(self.job.table)),
                    job=self.job,
                    up_stats=self.results.best_stats_by_level('up'),
                    is_up=is_up,
                    any_up=any_up,
                    down_stats=down_stats,
                    results=self.results))

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
