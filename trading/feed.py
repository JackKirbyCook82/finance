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
from datetime import datetime as Datetime
from datetime import date as Date
from pyalgotrade.bar import BasicBar, Frequency
from pyalgotrade.barfeed.membf import BarFeed as BarFeedBase


from utilities.files import ZIPCSVFile

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Feed", "StrategyBars", "HistoryBars", "Frequency"]
__copyright__ = "Copyright 2022, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class Feed(object):
    fields = ["ticker", "date", "datetime", "open", "close", "high", "low", "volume", "adjusted"]
    parsers = {"ticker": str, "date": lambda x: Date.strptime(x, "%Y/%m/%d"), "datetime": lambda x: Datetime.strptime(x, "%Y/%m/%d %H:%M:%S")}
    parser = float

    def __init_subclass__(cls, *args, dateformat="%Y/%m/%d", datetimeformat="%Y/%m/%d %H:%M:%S", **kwargs):
        cls.parsers.update({"date": lambda x: Datetime.combine(Date.strptime(x, dateformat), Datetime.min.time()), "datetime": lambda x: Datetime.strptime(x, datetimeformat)})

    def __init__(self, directory, filename):
        self.__directory = directory
        self.__filename = filename

    def __call__(self, ticker=None, starttime=None, stoptime=None):
        assert isinstance(starttime, (Datetime, type(None))) and isinstance(stoptime, (Datetime, type(None)))
        check_ticker = lambda x: str(x).upper() == str(ticker).upper() if x is not None else True
        check_starttime = lambda t: t <= starttime if t is not None else True
        check_stoptime = lambda t: t >= stoptime if t is not None else True
        criteria = lambda x, t: all([check_ticker(x), check_starttime(t), check_stoptime(t)])
        for content in iter(self):
            if criteria(content["ticker"], content["datetime"]):
                yield content

    def __iter__(self):
        with ZIPCSVFile(self.__directory, self.__filename, mode="r") as zfile:
            reader = zfile(fields=self.__class__.fields, parsers=self.__class__.parsers, parser=self.__class__.parser)
            for row in reader:
                row = {key: value for key, value in row.items() if value is not None}
                row["index"] = row.get("datetime", row.get("date", None))
                row.pop("date", None)
                row.pop("datetime", None)
                assert row["index"] is not None
                yield row


class BarsMeta(ABCMeta):
    fields = ["open", "high", "low", "close", "volume", "adjusted"]

    def __call__(cls, reader, *args, ticker=None, starttime=None, stoptime=None, **kwargs):
        assert isinstance(reader, Feed)
        assert isinstance(starttime, Datetime) and isinstance(stoptime, Date)
        feed = cls.feed(reader, ticker, starttime, stoptime)
        bars = cls.bars(reader, ticker, starttime, stoptime)
        instance = super(BarsMeta, cls).__call__(*args, feed=feed, bars=bars, fields=cls.fields, **kwargs)
        return instance

    def feed(cls, reader, ticker, starttime, stoptime):
        for content in reader(ticker=ticker, starttime=starttime, stoptime=stoptime):
            key = content["ticker"]
            index = content["index"]
            values = [content[field] for field in cls.fields]
            yield key, index, values

    def bars(cls, reader, ticker, starttime, stoptime):
        feed = cls.feed(reader, ticker, starttime, stoptime)
        bars = {}
        for key, index, values in iter(feed):
            if key not in bars.keys():
                bars[key] = []
            bars[key].append([index] + list(values))
        for key, values in bars.items():
            yield key, values


class StrategyBars(BarFeedBase, metaclass=BarsMeta):
    def __init__(self, *args, bars, frequency, length=None, **kwargs):
        assert frequency in Frequency.__members__
        super().__init__(frequency, length)
        for key, values in iter(bars):
            self.addBarsFromSequence(key, BasicBar(*values, frequency))

    def price(self, ticker): return self[ticker].getPriceDataSeries()
    def open(self, ticker): return self[ticker].getOpenDataSeries()
    def close(self, ticker): return self[ticker].getCloseDataSeries()
    def high(self, ticker): return self[ticker].getHighDataSeries()
    def low(self, ticker): return self[ticker].getLowDataSeries()
    def volume(self, ticker): return self[ticker].getVolumeDataSeries()
    def adjusted(self, ticker): return self[ticker].getAdjCloseDataSeries()

    @staticmethod
    def barsHaveAdjClose():
        return True


class HistoryBars(dict, metaclass=BarsMeta):
    def __init__(self, *args, bars, fields, **kwargs):
        super().__init__({key: pd.DataFrame(values, columns=["index", *fields]).set_index("index", inplace=False, drop=True) for key, values in iter(bars)})

    def __contains__(self, ticker): return ticker in self.__bars.keys()
    def __setitem__(self, ticker, dataframe): self.__bars[ticker].update(dataframe.set_index("index", inplace=False, drop=True), overwrite=True)
    def __getitem__(self, ticker): return self.__bars[ticker]

    def index(self, ticker): return self[ticker].index.to_series().drop(inplace=False)
    def price(self, ticker): return self[ticker]["adjusted"] if self.barsHaveAdjClose() else self[ticker]["close"]
    def open(self, ticker): return self[ticker]["open"]
    def close(self, ticker): return self[ticker]["close"]
    def high(self, ticker): return self[ticker]["high"]
    def low(self, ticker): return self[ticker]["low"]
    def volume(self, ticker): return self[ticker]["volume"]
    def adjusted(self, ticker): return self[ticker]["adjusted"]

    @staticmethod
    def barsHaveAdjClose():
        return True



