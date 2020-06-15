import pylab as pl
import pandas as pd
import numpy as np
from sys import stderr
from time import time
from contextlib import contextmanager
from arsenal.humanreadable import htime
from arsenal.terminal import colors
from arsenal.misc import ddict


def timers(title=None):
    return Benchmark(title)


class Benchmark(object):
    def __init__(self, title):
        self.title = title
        self.timers = ddict(Timer)
    def __getitem__(self, name):
        return self.timers[name]
    def compare(self, statistic=np.median):
        best = min(self.timers.values(), key=lambda t: statistic(t.times))
        for name in sorted(self.timers):
            if name != best.name:
                best.compare(self.timers[name])
    def values(self):
        return list(self.timers.values())
    def keys(self):
        return list(self.timers.keys())
    def items(self):
        return list(self.timers.items())
    def __len__(self):
        return len(self.timers)
    def __iter__(self):
        return iter(sorted(self.keys()))
    def plot_feature(self, feature, timecol='timer', ax=None, **kw):
        if ax is None: ax = pl.figure().add_subplot(111)
        for t in list(self.timers.values()):
            t.plot_feature(feature=feature,
                           timecol=timecol,
                           ax=ax,
                           **kw)
        if self.title is not None: ax.set_title(self.title)
        ax.legend(loc=2)
        return ax

    def plot_survival(self,*args,**kwargs):
        "Show the probability each algorithm is still running."
        for _, t in sorted(self.items()):
            t.plot_survival(*args,**kwargs)

    def run(self, methods, reps):
        from arsenal import iterview, restore_random_state
        if isinstance(methods, (tuple, list)):
            methods = {m.__name__: m for m in methods}

        jobs = [
            (name, seed)
            for seed in range(reps)   # TODO: use a better strategy for picking random seeds.
            for name in methods
        ]
        np.random.shuffle(jobs)       # shuffle jobs to avoid weird ordering correlations
        for name, seed in iterview(jobs):
            with restore_random_state(seed):
                with self[name]:
                    methods[name]()


class Timer(object):
    """
    >>> from time import sleep
    >>> a = Timer('A')
    >>> b = Timer('B')
    >>> with a:
    ...     sleep(0.5)
    >>> with b:
    ...     sleep(1)
    >>> a.compare(b)          #doctest:+SKIP
    A is 2.0018x faster

    """
    def __init__(self, name=None):
        self.name = name
        self.times = []
        self.features = []
        self.b4 = None

    def __enter__(self):
        self.b4 = time()

    def __exit__(self, *_):
        self.times.append(time() - self.b4)

    def __str__(self):
        return 'Timer(name=%s, avg=%g, std=%g)' % (self.name, self.avg, self.std)

    def __call__(self, **features):
        self.features.append(features)
        return self

    @property
    def avg(self):
        return np.mean(self.times)

    @property
    def median(self):
        return np.median(self.times)

    @property
    def std(self):
        if len(self.times) <= 1:
            return 0.0
        return np.std(self.times, ddof=1)

    @property
    def total(self):
        return sum(self.times)

    def compare(self, other, attr='median', verbose=True):
        if len(self.times) == 0 or len(other.times) == 0:
            print('%s %s %s' % (self.name, '???', other.name))
            return
        if getattr(self, attr) <= getattr(other, attr):
            print('%s is %6.4fx faster than %s %s' \
                % (self.name,
                   getattr(other, attr) / getattr(self, attr),
                   other.name,
                   ((colors.yellow % '(%s: %s: %g %s: %g)' % (attr,
                                                      other.name, getattr(other, attr),
                                                      self.name, getattr(self, attr)))
                    if verbose else '')
                ))
        else:
            other.compare(self, attr=attr, verbose=verbose)

#    def compare_many(self, *others, **kw):
#        for x in sorted(others, key=lambda x: x.name):
#            if x != self:
#                self.compare(x, **kw)

    def plot_feature(self, feature, timecol='timer', ax=None, **kw):
        if ax is None: ax = pl.figure().add_subplot(111)
        df = self.dataframe(timecol)
        a = df.groupby(feature).median()

        X = a.index
        Y = a[timecol]
        ax.set_xlabel(feature)
        ax.set_ylabel('time (seconds)')

        if 'label' not in kw:
            # use name of the timer as default label.
            kw['label'] = f'{self.name}'

        [line] = ax.plot(X, Y, lw=2, alpha=0.5, **kw)
        del kw['label'] # delete label so it doesn't appear twice in the legend

        c = line.get_color()
        ax.scatter(X, Y, c=c, lw=0, label=None, alpha=0.25, **kw)
        #ax.scatter(df[feature], df[timecol], c=c, alpha=0.25, marker='.', label=None, **kw)

        data = []
        for f, dd in df.groupby(feature):
            data.append([
                f,
                np.percentile(dd[timecol], 20),
                np.percentile(dd[timecol], 80),
            ])
        data = list(sorted(data))
        fs, ls, us = zip(*data)
        ax.fill_between(fs, ls, us, alpha=0.25, color=c)

        #elif 'box' in show:
        #    # TODO: doen't work very well yet. need to fill out the x-axis since
        #    # feature might not be dense. Should throw an error if feature isn't
        #    # integral.
        #    ddd = [np.asarray(dd[timecol]) for f, dd in sorted(df.groupby(feature))]
        #    ax.boxplot(ddd)

        return ax

    def dataframe(self, timecol='timer'):
        df = pd.DataFrame(list(self.features))
        df[timecol] = self.times
        return df

    def filter(self, f, name=None):
        t = Timer(name)
        t.times, t.features = list(zip(*[(x,y) for (x,y) in zip(self.times, self.features) if f(x,y)]))
        return t

    def bucket_filter(self, feature_to_bucket, bucket_filter):
        df = self.dataframe()
        data = []
        for k, d in df.groupby(feature_to_bucket):
            d = d[bucket_filter(k, d)]
            _, x = list(zip(*d.iterrows()))
            data.append(x)
        return pd.DataFrame(data)

    def trim_slow(self, feature_to_bucket, threshold):
        return self.bucket_filter(
            feature_to_bucket,
            lambda k, d: d.timer <= d.timer.quantile(threshold)
        )

    def plot_survival(self, ax=None):
        if ax is None: ax = pl.figure().add_subplot(111)
        from arsenal.maths import cdf
        ts = np.array(self.times)
        xs = np.linspace(0, ts.max(), 1000)
        ax.plot(xs, 1-cdf(ts)(xs), label=self.name)
        ax.legend(loc='best'); ax.set_xscale('log'); ax.set_yscale('log')


@contextmanager
def timeit(name, fmt='{name} ({htime})', header=None):
    """Context Manager which prints the time it took to run code block."""
    if header is not None:
        print(header)
    b4 = time()
    yield
    sec = time() - b4
    if sec < 60:
        ht = '%.4f sec' % sec
    else:
        ht = htime(sec)
    print(fmt.format(name=name, htime=ht, sec=sec), file=stderr)


def main():
    from arsenal.iterview import iterview
    from time import sleep
    from numpy.random import uniform
    t = Timer('test')

    for i in iterview(range(1, 20)):
        for _ in range(10):
            with t(i=i):
                c = 0.01
                z = max(i**2 * 0.0001 + uniform(-c, c), 0.0)
                sleep(z)

    a = t.plot_feature('i')
    print(a)
    pl.show()


if __name__ == '__main__':
    main()
