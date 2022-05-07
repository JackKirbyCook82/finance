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
__all__ = ["Strategy", "Volatility", "SMA", "Total", "Value"]
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
class Value(EventBasedSeries): pass


class Strategy(BacktestingStrategy, ABC):
    def __init__(self, feed, cash, *args, **kwargs):
        super().__init__(feed, cash)
        self.setUseAdjustedValues(False)
        self.__arguments = tuple()
        self.__parameters = dict()
        self.__price = None
        self.__open = None
        self.__close = None
        self.__high = None
        self.__low = None
        self.__volume = None
        self.__adjusted = None

    def __call__(self, *args, **kwargs):
        self.__arguments = args
        self.__parameters = kwargs

    def start(self): self.run()
    def stop(self): super().stop()

    def onBars(self, bars):
        self.__price = lambda x: bars[x].getPrice()
        self.__open = lambda x: bars[x].getOpen()
        self.__close = lambda x: bars[x].getClose()
        self.__high = lambda x: bars[x].getHigh()
        self.__low = lambda x: bars[x].getLow()
        self.__volume = lambda x: bars[x].getVolume()
        self.__adjusted = lambda x: bars[x].getAdjClose()
        self.execute(*self.__arguments, **self.__parameters)

    def current(self): return self.getCurrentDateTime()
    def price(self, ticker): return self.__price(ticker)
    def open(self, ticker): return self.__open(ticker)
    def close(self, ticker): return self.__close(ticker)
    def high(self, ticker): return self.__high(ticker)
    def low(self, ticker): return self.__low(ticker)
    def volume(self, ticker): return self.__volume(ticker)
    def adjusted(self, ticker): return self.__adjusted(ticker)

    @abstractmethod
    def execute(self, *args, **kwargs): pass







