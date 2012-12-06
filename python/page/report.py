

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

        self.env = Environment(loader=PackageLoader('page'))

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
        self.cached_stat_hists = None

    def make_report(self):
        cwd = os.getcwd()
        try:
            os.chdir(self.output_dir)
            self.make_jinja_report()
        finally:
            os.chdir(cwd)

    def stat_hists(self):

        """Plot histograms showing the distribution of statistics, for
        each combination of direction (up/down), confidence level, and
        class.

        Returns a (direction x level x class) array giving the paths
        to the resulting plots.

        """
        if self.cached_stat_hists is not None:
            return self.cached_stat_hists

        fmt = 'stat_hist_direction_{direction}_level_{level}_class_{cls}.png'

        results = self.results
        res = np.empty((2, results.num_levels, results.num_classes), 
                       dtype=object)

        print "Plotting histograms"

        directions = ['up', 'down']

        dir_stats = [
            results.best_up_stats_by_level,
            results.best_down_stats_by_level
            ]

        dir_cutoffs = [
            results.cutoffs_by_level('up'),
            results.cutoffs_by_level('down'),
            ]

        xmax = max(np.max(dir_stats[0]),
                   np.max(dir_stats[1]))
        xmin = min(np.min(dir_stats[0]),
                   np.min(dir_stats[1]))

        for idx in np.ndindex(np.shape(res)):

            (d, i, c) = idx
            filename = fmt.format(direction=d, level=i, cls=c)
            res[d, i, c] = filename

            if os.path.exists(filename):
                print "{filename} already exists, skipping".format(
                    filename=filename)
                continue

            stats = dir_stats[d]

            cutoffs = dir_cutoffs[d]

            plt.cla()
            plt.hist(stats[i, c], bins=50)
            plt.axvline(x=cutoffs[i, c])
            plt.xlim([xmin, xmax])
            plt.xlabel('Statistic')
            plt.ylabel('Features')
            plt.title("Distribution of statistic\nClass \"{cls}\" {direction}-regulated at {conf:.1%} conf.".format(
                    direction=['up', 'down'][d],
                    cls=self.job.condition_names[c],
                    conf=self.results.conf_levels[i]))
            plt.savefig(filename) 

        self.cached_stat_hists = res
        return res


    def hists_by_test_and_class(self):
        print "Plotting histograms"
        results = self.results
        stats   = self.stats
        maxstat = np.max(stats)
        minstat = np.min(stats)
        stat_hists = []

        for i in range(results.num_tests):
            for c in range(1, results.num_classes):
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
        for c in range(1, results.num_classes):
            with open('stat_hist_class_{cls}.html'.format(cls=c), 'w') as out:
                template = self.env.get_template('stat_hist_class.html')
                stat_hists = [
                    'stat_hist_test_{test}_class_{cls}.png'.format(
                        test=test, cls=c)
                    for test in range(results.num_tests)]
                out.write(template.render(
                        job=self.job,
                        stat_hists=stat_hists,
                        condition_num=c,
                        ))
        return stat_hists

    def make_index(self):
        results = self.results
        colors = matplotlib.rcParams['axes.color_cycle']
        plt.clf()
        for c in range(1, results.num_classes):
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
            template = self.env.get_template('index.html')
        
            out.write(template.render(
                    condition_nums=range(1, results.num_classes),
                    results=self.results,
                    job=self.job,
                    levels=results.conf_levels))

    def make_jinja_report(self):

        stats = self.stats
        output_dir = self.output_dir

        stat_hists = []
    
        raw_conf_plots = []

        results = self.results

        if self.plot_histograms:
            stat_hists = self.hists_by_test_and_class()

        self.make_index()
        self.conf_detail_pages()

    def conf_detail_pages(self):

        results = self.results

        ##
        ## Make the detail page for each level
        ##
            
        up_cutoffs   = results.up_cutoffs_by_level
        down_cutoffs = results.down_cutoffs_by_level
        feature_to_up_conf = results.feature_to_conf_by_conf('up')
        feature_to_down_conf = results.feature_to_conf_by_conf('down')
        up_stats   = results.best_stats_by_level('up')
        down_stats = results.best_stats_by_level('down')

        any_regulated  = np.zeros((results.num_levels, results.num_features), int)

        determination = np.zeros((results.num_levels, 
                                  results.num_classes,
                                  results.num_features), dtype=int)

        feature_to_conf = np.zeros((results.num_levels,
                                    results.num_classes,
                                    results.num_features))
        
        feature_to_stat = np.zeros((results.num_levels,
                                    results.num_classes,
                                    results.num_features))


        hists = self.stat_hists()

        for i in range(results.num_levels):
            for j in range(results.num_features):
                for c in range(1, results.num_classes):

                    if up_stats[i, c, j] >= up_cutoffs[i, c]:
                        determination[i, c, j] = 1
                        feature_to_stat[i, c, j] = up_stats[i, c, j]
                        feature_to_conf[i, c, j] = feature_to_up_conf[i, c, j]
                    elif down_stats[i, c, j] <= -down_cutoffs[i, c]:
#                        print "Down stats is " + str(down_stats[i, c, j]) + ", cutoff is " + str(-down_cutoffs[i, c])
                        determination[i, c, j] = 2
                        feature_to_stat[i, c, j] = down_stats[i, c, j]
                        feature_to_conf[i, c, j] = feature_to_down_conf[i, c, j]

                any_regulated[i, j] = np.any(determination[i, :, j] > 0)

            with open('conf_level_detail_' + str(i) + '.html', 'w') as out:
                template = self.env.get_template('features_by_confidence.html')
                out.write(
                    template.render(
                        level=i,
                        hists=hists,
                        condition_nums=range(1, results.num_classes),
                        job=self.job,
                        results=self.results,
                        feature_nums=range(results.num_features),

                        any_determination=any_regulated,

                        feature_to_stat=feature_to_stat,
                        feature_to_conf=feature_to_conf,

                        up_stats=up_stats,
                        feature_to_up_conf=feature_to_up_conf,

                        down_stats=down_stats,
                        determination=determination,
                        feature_to_down_conf=feature_to_down_conf,

))

