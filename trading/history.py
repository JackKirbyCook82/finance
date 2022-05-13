# -*- coding: utf-8 -*-
"""
Created on Mon May 9 2022
@name:   Trading History Objects
@author: Jack Kirby Cook

"""

import warnings
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["History", "HistoryIndicators"]
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


class Volatility(HistoryBasedFilter, reduction=stdev, function=var): pass
class SMA(HistoryBasedFilter, reduction=average): pass
class Total(HistoryBasedFilter, reduction=total): pass


class HistoryIndicators(object):
    VOLATILITY = Volatility
    SMA = SMA
    TOTAL = Total


class MissingHistorySeriesError(Exception): pass
class ExistingHistorySeriesError(Exception): pass


class HistoryProxyMeta(type):
    def __call__(cls, ticker, feed):
        series = {key: getattr(feed[ticker], key) for key in cls.fields}
        instance = super(HistoryProxyMeta, cls).__call__(ticker, series)
        return instance


class HistoryProxy(object, metaclass=HistoryProxyMeta):
    def __init__(self, ticker, series):
        self.__ticker = ticker
        self.__series = series

    def __contains__(self, attr):
        return attr in self.__series.keys()

    def __getattr__(self, attr):
        if attr not in self:
            raise MissingHistorySeriesError(attr)
        return self.__series[attr]

    def __setattr__(self, attr, series):
        assert isinstance(series, pd.Series)
        if attr in self:
            raise ExistingHistorySeriesError(attr)
        self.__series[attr] = series


class History(ABC):
    def __init__(self, feed, *args, **kwargs):
        self.__feed = feed
        self.__proxys = {}
        self.setup(*args, **kwargs)

    def __getitem__(self, ticker):
        try:
            return self.__proxys[ticker]
        except IndexError:
            self.__proxys[ticker] = HistoryProxy(ticker, self.__feed)
            return self.__proxys[ticker]

    def __call__(self, *args, **kwargs):
        self.execute(*args, **kwargs)
        return

    @abstractmethod
    def setup(self, *args, **kwargs): pass
    @abstractmethod
    def execute(self, *args, **kwargs): pass












