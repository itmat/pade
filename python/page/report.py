

import matplotlib
import shutil
import matplotlib.pyplot as plt
import numpy as np
import os
from jinja2 import Environment, PackageLoader



class Report:

    def __init__(self, job, output_dir, results):

        self.env = Environment(loader=PackageLoader('page'))

        self.job           = job
        self.results       = results
        self.output_dir    = output_dir
        self.stats         = results.stats
        self.conf_levels   = results.conf_levels

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

        dir_stats = results.best_stats_by_level
        dir_cutoffs = results.cutoffs_by_level

        xmax = np.max(dir_stats)
        xmin = np.min(dir_stats)

        for idx in np.ndindex(np.shape(res)):

            (d, i, c) = idx
            filename = fmt.format(direction=d, level=i, cls=c)
            res[idx] = filename

            if os.path.exists(filename):
                print "{filename} already exists, skipping".format(
                    filename=filename)
                continue

            plt.cla()
            plt.hist(dir_stats[idx], bins=50)
            plt.axvline(x=dir_cutoffs[idx])
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
                        results=self.results
                        ))
        return stat_hists

    def make_index(self):
        results = self.results
        colors = matplotlib.rcParams['axes.color_cycle']
        plt.clf()
        for c in range(1, results.num_classes):
            plt.plot(self.results.conf_levels,
                     self.results.best_counts[0, :, c],
                     colors[c] + '-^',
                     label=self.job.condition_names[c] + ' up')
            plt.plot(self.results.conf_levels,
                     self.results.best_counts[1, :, c],
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

        if self.plot_histograms:
            self.hists_by_test_and_class()

        self.make_index()
        self.conf_detail_pages()

        src = os.path.join(os.path.dirname(__file__),
                           '996grid/code/css')

        shutil.rmtree('css', True)
        shutil.copytree(src, 'css')

        with open('css/custom.css', 'w') as out:
            template = self.env.get_template('custom.css')
            out.write(template.render())

    def conf_detail_pages(self):

        results = self.results

        shape = (results.num_levels, 
                 results.num_classes,
                 results.num_features)

        up = 0
        down = 1

        cutoffs = results.cutoffs_by_level
        stats = results.best_stats_by_level
        feature_to_conf_by_dir = results.feature_to_conf_by_conf
        any_regulated  = np.zeros((results.num_levels, results.num_features), int)
        determination   = np.zeros(shape, dtype=int)
        feature_to_conf = np.zeros(shape)        
        feature_to_stat = np.zeros(shape)

        hists = self.stat_hists()

        for idx in np.ndindex(shape):
            (i, c, j) = idx

            up_stat = stats[(up,) + idx]
            down_stat = stats[(down,) + idx]

            if up_stat >= cutoffs[up, i, c]:
                determination[idx] = 1
                feature_to_stat[idx] = up_stat
                feature_to_conf[idx] = feature_to_conf_by_dir[(up,)+idx]

            elif down_stat <= -cutoffs[down, i, c]:
                determination[idx] = 2
                feature_to_stat[idx] = down_stat
                feature_to_conf[idx] = feature_to_conf_by_dir[(down,)+idx]

        for i, j in np.ndindex(np.shape(any_regulated)):
            any_regulated[i, j] = np.any(determination[i, :, j] > 0)

        for i in range(results.num_levels):
            with open('conf_level_detail_' + str(i) + '.html', 'w') as out:
                template = self.env.get_template('features_by_confidence.html')
                out.write(
                    template.render(
                        level=i,
                        hists=hists,
                        job=self.job,
                        results=self.results,
                        any_determination=any_regulated,
                        feature_to_stat=feature_to_stat,
                        feature_to_conf=feature_to_conf,
                        determination=determination,
                        ))

