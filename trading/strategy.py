# -*- coding: utf-8 -*-
"""
Created on Weds Apr 27 2022
@name:   Trading Strategy Objects
@author: Jack Kirby Cook

"""

import warnings
import logging
import numpy as np
from abc import ABC, abstractmethod
from pyalgotrade.utils import collections
from pyalgotrade.dataseries import SequenceDataSeries
from pyalgotrade.strategy import BacktestingStrategy

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Strategy", "StrategyIndicators"]
__copyright__ = "Copyright 2022, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


diff = lambda x: np.diff(x)
var = lambda x: np.diff(x) / x[1:] * 100

average = lambda x: np.average(x)
total = lambda x: np.sum(x)
maximum = lambda x: np.max(x)
minimum = lambda x: np.min(x)
first = lambda x: x[0]
last = lambda x: x[-1]
stdev = lambda x: np.std(x)


class EventWindow(object):
    def __init_subclass__(cls, *args, reduction, **kwargs):
        cls.reduction = reduction

    def __init__(self, period):
        assert(period > 0)
        assert(isinstance(period, int))
        self.__values = collections.NumPyDeque(period, None)
        self.__period = period

    def onNewValue(self, key, value): self.__values.append(value)
    def getValues(self): return self.__values.data()
    def getWindowSize(self): return self.__period
    def windowFull(self): return len(self.__values) == self.__period
    def getValue(self): return self.__class__.reduction(self.getValues())


class EventBasedFilter(SequenceDataSeries):
    def __init_subclass__(cls, *args, reduction, function=lambda x: x, **kwargs):
        cls.window = type("{}Window".format(cls.__name__), (EventWindow,), {}, reduction=reduction)
        cls.function = function

    def __init__(self, dataseries, period, size=None):
        super(EventBasedFilter, self).__init__(size)
        self.__eventwindow = self.__class__.window(period)
        self.__dataseries = dataseries
        self.__dataseries.getNewValueEvent().subscribe(self.onNewValue)

    def __int__(self): return np.int(self[-1]) if self[-1] is not None else None
    def __float__(self): return np.float(self[-1]) if self[-1] is not None else None

    def onNewValue(self, dataseries, key, value):
        self.__eventWindow.onNewValue(key, value)
        newvalue = self.__eventWindow.getValue()
        newvalue = self.__class__.function(newvalue)
        self.appendWithDateTime(key, newvalue)

    def getDataSeries(self): return self.__dataseries
    def getEventWindow(self): return self.__eventwindow


class EventBasedSeries(SequenceDataSeries):
    def __init_subclass__(cls, *args, function=lambda x: x, **kwargs):
        cls.function = function

    def __init__(self, dataseries, size=None):
        super(EventBasedSeries, self).__init__(size)
        self.__dataseries = dataseries
        self.__dataseries.getNewValueEvent().subscribe(self.onNewValue)

    def __int__(self): return np.int(self[-1]) if self[-1] is not None else None
    def __float__(self): return np.float(self[-1]) if self[-1] is not None else None

    def onNewValue(self, dataseries, key, value):
        newvalue = self.__class__.function(value)
        self.appendWithDateTime(key, newvalue)

    def getDataSeries(self): return self.__dataseries


class Volatility(EventBasedFilter, reduction=stdev, function=var): pass
class SMA(EventBasedFilter, reduction=average): pass
class Total(EventBasedFilter, reduction=total): pass


class StrategyIndicators(object):
    VOLATILITY = Volatility
    SMA = SMA
    TOTAL = Total


class MissingStrategySeriesError(Exception): pass
class ExistingStrategySeriesError(Exception): pass


class StrategyProxyMeta(type):
    fields = ["open", "close", "high", "low", "volume", "adjusted"]

    def __call__(cls, ticker, feed):
        series = {key: getattr(feed[ticker], value)() for key, value in cls.fields.items()}
        instance = super(StrategyProxyMeta, cls).__call__(ticker, series)
        return instance


class StrategyProxy(object, metaclass=StrategyProxyMeta):
    def __init__(self, ticker, series):
        self.__ticker = ticker
        self.__series = series

    def __contains__(self, attr):
        return attr in self.__series.keys()

    def __getattr__(self, attr):
        if attr not in self:
            raise MissingStrategySeriesError(attr)
        return self.__series[attr]

    def __setattr__(self, attr, series):
        assert isinstance(series, SequenceDataSeries)
        if attr in self:
            raise ExistingStrategySeriesError(attr)
        self.__series[attr] = series


class Strategy(BacktestingStrategy, ABC):
    def __init__(self, feed, cash, *args, **kwargs):
        super().__init__(feed, cash)
        self.__feed = feed
        self.__proxys = {}
        self.__arguments = tuple()
        self.__parameters = dict()
        self.setUseAdjustedValues(False)
        self.setup(*args, **kwargs)

    def __getitem__(self, ticker):
        try:
            return self.__proxys[ticker]
        except IndexError:
            self.__proxys[ticker] = StrategyProxy(ticker, self.__feed)
            return self.__proxys[ticker]

    def __call__(self, *args, **kwargs):
        self.__arguments = args
        self.__parameters = kwargs
        self.run()
        return

    def onBars(self, bars):
        self.execute(*self.__arguments, **self.__parameters)

    @abstractmethod
    def setup(self, *args, **kwargs): pass
    @abstractmethod
    def execute(self, *args, **kwargs): pass







