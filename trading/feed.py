# -*- coding: utf-8 -*-
"""
Created on Weds Apr 27 2022
@name:   Trading Feed Objects
@author: Jack Kirby Cook

"""

import warnings
import logging
import pandas as pd
from abc import ABCMeta
from datetime import date as Date
from datetime import datetime as Datetime
from collections import OrderedDict as ODict
from collections import namedtuple as ntuple
from pyalgotrade.bar import BasicBar, Frequency
from pyalgotrade.barfeed.membf import BarFeed as BarFeedBase

from files.csvs import CSVFile
from utilities.parsers import datetimeparser

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["BarReader", "StrategyFeed", "HistoryFeed", "Frequency"]
__copyright__ = "Copyright 2022, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class BarReader(object):
    def __init__(self, directory, file):
        self.__parsers = {"ticker": str, "date": datetimeparser, "open": float, "close": float, "high": float, "low": float, "volume": int, "adjusted": float}
        self.__directory = directory
        self.__file = file

    def __call__(self, ticker=None, starttime=None, stoptime=None):
        assert isinstance(starttime, (Datetime, type(None))) and isinstance(stoptime, (Datetime, type(None)))
        check_ticker = lambda x: str(x).upper() == str(ticker).upper() if x is not None else True
        check_starttime = lambda t: t <= starttime if t is not None else True
        check_stoptime = lambda t: t >= stoptime if t is not None else True
        criteria = lambda x, t: all([check_ticker(x), check_starttime(t), check_stoptime(t)])
        for content in iter(self):
            if criteria(content["ticker"], content["date"]):
                yield content

    def __iter__(self):
        with CSVFile(directory=self.directory, file=self.file, mode="r", fields=self.fields) as reader:
            for row in iter(reader):
                row = {key: self.parsers[key](value) for key, value in row.items() if key in self.fields and value is not None}
                yield row

    @property
    def directory(self): return self.__directory
    @property
    def file(self): return self.__file
    @property
    def fields(self): return list(self.__parsers.values())
    @property
    def parsers(self): return self.__parsers


class BarFeedProxy(ntuple("BarFeedProxy", ["open", "close", "high", "low", "volume", "adjusted"])):
    pass


class BarFeedMeta(ABCMeta):
    fields = ["open", "close", "high", "low", "volume", "adjusted"]

    def __call__(cls, reader, *args, ticker=None, starttime=None, stoptime=None, **kwargs):
        assert isinstance(reader, BarReader)
        assert isinstance(starttime, Datetime) and isinstance(stoptime, Date)
        feed = cls.feed(reader, ticker, starttime, stoptime, fields=cls.fields)
        instance = super(BarFeedMeta, cls).__call__(*args, feed=feed, **kwargs)
        return instance

    @staticmethod
    def feed(reader, ticker, starttime, stoptime, *args, fields=[], **kwargs):
        for content in reader(ticker=ticker, starttime=starttime, stoptime=stoptime):
            ticker = content["ticker"]
            date = content["date"]
            contents = ODict([(field, content[field]) for field in fields])
            yield ticker, date, contents


class StrategyFeed(BarFeedBase, metaclass=BarFeedMeta):
    def __init__(self, *args, feed, frequency, length=None, **kwargs):
        assert frequency in Frequency.__members__
        super().__init__(frequency, length)
        for ticker, date, contents in iter(feed):
            bar = BasicBar([date] + list(contents.values()), frequency)
            self.addBarsFromSequence(ticker, bar)

    def __getitem__(self, ticker):
        attrs = ["getOpenDataSeries", "getCloseDataSeries", "getHighDataSeries", "getLowDataSeries", "getVolumeDataSeries", "getAdjCloseDataSeries"]
        return BarFeedProxy(*[getattr(super().__getitem__(ticker), attr) for attr in attrs])

    @staticmethod
    def barsHaveAdjClose(): return True


class HistoryFeed(dict, metaclass=BarFeedMeta):
    def __init__(self, *args, feed, **kwargs):
        records = [{"ticker": ticker, "date": date, **contents} for ticker, date, contents in iter(feed)]
        groups = pd.DataFrame(records).set_index("date", drop=True, inplace=False).groupby("ticker")
        super().__init__({ticker: dataframe for ticker, dataframe in iter(groups)})

    def __getitem__(self, ticker):
        columns = ["open", "close", "high", "low", "volume", "adjusted"]
        return BarFeedProxy(*[lambda: super().__getitem__(ticker)[column] for column in columns])

    @staticmethod
    def barsHaveAdjClose(): return True



