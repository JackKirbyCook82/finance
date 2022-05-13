# -*- coding: utf-8 -*-
"""
Created on Weds Apr 27 2022
@name:   Trading Strategy Objects
@author: Jack Kirby Cook

"""

import warnings
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from pyalgotrade.utils import collections
from pyalgotrade.dataseries import SequenceDataSeries
from pyalgotrade.strategy import BacktestingStrategy

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Strategy", "History", "Indicators"]
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


class HistoryBasedFilter(pd.Series):
    def __init_subclass__(cls, *args, reduction, function=lambda x: x, **kwargs):
        cls.reduction = reduction
        cls.function = function

    def __new__(cls, series, period):
        assert(period > 0)
        assert(isinstance(period, int))
        assert isinstance(series, pd.Series)
        reduction = cls.reduction
        function = lambda x: cls.function(x) if not pd.isna(x) else x
        series = series.rolling(period).apply(reduction)
        return series.apply(function).dropna(inplace=False)


class EventBasedSeries(pd.Series):
    def __init_subclass__(cls, *args, function=lambda x: x, **kwargs):
        cls.function = function

    def __new__(cls, series):
        assert isinstance(series, pd.Series)
        function = cls.function
        return series.apply(function).dropna(inplace=False)


class Indicators:
    class Strategy:
        class Volatility(EventBasedFilter, reduction=stdev, function=var): pass
        class SMA(EventBasedFilter, reduction=average): pass
        class Total(EventBasedFilter, reduction=total): pass

    class History:
        class Volatility(HistoryBasedFilter, reduction=stdev, function=var): pass
        class SMA(HistoryBasedFilter, reduction=average): pass
        class Total(HistoryBasedFilter, reduction=total): pass


class BarProxyMeta(type):
    fields = ["open", "close", "high", "low", "volume", "adjusted"]

    def __call__(cls, ticker, feed):
        contents = {field: getattr(feed[ticker], field)() for field in cls.fields}
        instance = super(BarProxyMeta, cls).__call__(contents)
        return instance


class BarProxy(dict, metaclass=BarProxyMeta):
    def __getattr__(self, attr): return self[attr]
    def __setattr__(self, attr, series): self[attr] = series


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
        if ticker in self.__proxys.keys():
            return self.__proxys[ticker]
        proxy = BarProxy(ticker, self.__feed)
        self.__proxy[ticker] = proxy
        return proxy

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


class History(ABC):
    def __init__(self, feed, *args, **kwargs):
        self.__feed = feed
        self.__proxys = {}
        self.setup(*args, **kwargs)

    def __getitem__(self, ticker):
        if ticker in self.__proxys.keys():
            return self.__proxys[ticker]
        proxy = BarProxy(ticker, self.__feed)
        self.__proxy[ticker] = proxy
        return proxy

    def __call__(self, *args, **kwargs):
        self.execute(*args, **kwargs)
        return

    @abstractmethod
    def setup(self, *args, **kwargs): pass
    @abstractmethod
    def execute(self, *args, **kwargs): pass




