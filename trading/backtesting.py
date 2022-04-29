# -*- coding: utf-8 -*-
"""
Created on Weds Apr 27 2022
@name:   Trading Backtesting Application
@author: Jack Kirby Cook

"""

import warnings
import logging
import numpy as np
from datetime import datetime as Datetime
from datetime import date as Date
from pyalgotrade.bar import BasicBar
from pyalgotrade.barfeed.membf import BarFeed
from pyalgotrade.utils import collections
from pyalgotrade.dataseries import SequenceDataSeries

from utilities.files import ZIPCSVFile

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Feed", "Volatility", "SMA", "Total", "Value"]
__copyright__ = "Copyright 2022, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


diff = lambda x: np.diff(x)
pctdiff = lambda x: np.diff(x) / x[1:] * 100

average = lambda x: np.average(x)
total = lambda x: np.sum(x)
maximum = lambda x: np.max(x)
minimum = lambda x: np.min(x)
first = lambda x: x[0]
last = lambda x: x[-1]
stdev = lambda x: np.std(x)


class Feed(BarFeed):
    fields = ["date", "datetime", "ticker", "open", "close", "high", "low", "adjusted"]
    order = ["open", "high", "low", "close", "volume", "adjusted"]
    parsers = {"ticker": str, "date": lambda x: Date.strptime(x, "%Y/%m/%d"), "datetime": lambda x: Datetime.strptime(x, "%Y/%m/%d %H:%M:%S")}
    parser = float

    def __init_subclass__(cls, *args, dateformat="%Y/%m/%d", datetimeformat="%Y/%m/%d %H:%M:%S", **kwargs):
        cls.parsers.update({"date": lambda x: Date.strptime(x, dateformat), "datetime": lambda x: Datetime.strptime(x, datetimeformat)})

    def __init__(self, directory, filename, frequency, length=None):
        super().__init__(frequency, length)
        bars = {}
        for key, values in self.generator(directory, filename):
            if key not in bars.keys():
                bars[key] = []
            bars[key].append(values)
        for key, values in bars.items():
            self.addBarsFromSequence(key, BasicBar(*values, frequency))

    def load(self, directory, filename):
        with ZIPCSVFile(directory, filename, mode="r") as zfile:
            reader = zfile(fields=self.__class__.fields, parsers=self.__class__.parsers, parser=self.__class__.parser)
            for row in reader:
                row = {key: value for key, value in row.items() if value is not None}
                key = row["ticker"]
                index = row["datetime"] if "datetime" in row.keys() else Datetime.combine(row["date"], Datetime.min.time())
                values = [index] + [row[field] for field in self.__class__.order]
                yield key, values

    def barsHaveAdjClose(self): return True


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

    def __int__(self): return np.int(self[-1])
    def __float__(self): return np.float(self[-1])

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

    def __int__(self): return np.int(self[-1])
    def __float__(self): return np.float(self[-1])

    def onNewValue(self, dataseries, key, value):
        newvalue = self.__class__.function(value)
        self.appendWithDateTime(key, newvalue)

    def getDataSeries(self): return self.__dataseries


class Volatility(EventBasedFilter, reduction=stdev, function=pctdiff): pass
class SMA(EventBasedFilter, reduction=average): pass
class Total(EventBasedFilter, reduction=total): pass
class Value(EventBasedSeries): pass



