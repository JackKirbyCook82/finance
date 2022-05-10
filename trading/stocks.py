# -*- coding: utf-8 -*-
"""
Created on Mon May 9 2022
@name:   Trading Stock Objects
@author: Jack Kirby Cook

"""

import warnings
import logging
import numpy as np

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = []
__copyright__ = "Copyright 2022, Jack Kirby Cook"
__license__ = ""

import pandas as pd

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


class HistoryBasedFilter(object):
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
        return series.apply(function)


class EventBasedSeries(object):
    def __init_subclass__(cls, *args, function=lambda x: x, **kwargs):
        cls.function = function

    def __new__(cls, series):
        assert isinstance(series, pd.Series)
        function = cls.function
        return series.apply(function)


class Volatility(HistoryBasedFilter, reduction=stdev, function=var): pass
class SMA(HistoryBasedFilter, reduction=average): pass
class Total(HistoryBasedFilter, reduction=total): pass


class StockHistory(object):
    def __init__(self, feed, *args, **kwargs):
        self.__feed = feed

    def index(self, ticker): return self.__feed.index(ticker)
    def price(self, ticker): return self.__feed.price(ticker)
    def open(self, ticker): return self.__feed.open(ticker)
    def close(self, ticker): return self.__feed.close(ticker)
    def high(self, ticker): return self.__feed.high(ticker)
    def low(self, ticker): return self.__feed.low(ticker)
    def volume(self, ticker): return self.__feed.volume(ticker)
    def adjusted(self, ticker): return self.__feed.adjusted(ticker)





