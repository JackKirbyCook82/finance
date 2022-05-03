# -*- coding: utf-8 -*-
"""
Created on Weds Apr 27 2022
@name:   Trading Feed Objects
@author: Jack Kirby Cook

"""

import warnings
import logging
from abc import ABCMeta
from datetime import datetime as Datetime
from datetime import date as Date
from pyalgotrade.bar import BasicBar, Frequency
from pyalgotrade.barfeed.membf import BarFeed as BarFeedBase


from utilities.files import ZIPCSVFile

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Feed", "BarFeed", "Frequency"]
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
                row["datetime"] = row.get("datetime", row.get("date", None))
                row.pop("date", None)
                assert row["datetime"] is not None
                yield row


class BarFeedMeta(ABCMeta):
    def __init__(cls, *args, **kwargs):
        cls.__dateformat = kwargs.get("dateformat", getattr(cls, "dateformat", "%Y/%m/%d"))
        cls.__datetimeformat = kwargs.get("datetimeformat", getattr(cls, "datetimeformat", "%Y/%m/%d %H:%M:%S"))

    def __call__(cls, *args, starttime, stoptime, **kwargs):
        starttime, stoptime = cls.parsers[type(starttime)](starttime), cls.parsers[type(stoptime)](stoptime)
        instance = super(BarFeedMeta, cls).__call__(*args, starttime=starttime, stoptime=stoptime, **kwargs)
        return instance

    @property
    def parsers(cls): return {str: lambda x: Datetime.strptime(x, cls.datetimeformat), Date: lambda x: Datetime.combine(x, Datetime.min.time()), Datetime: lambda x: x}
    @property
    def dateformat(cls): return cls.__dateformat
    @property
    def datetimeformat(cls): return cls.__datetimeformat
    @dateformat.setter
    def dateformat(cls, dateformat): cls.__dateformat = dateformat
    @datetimeformat.setter
    def datetimeformat(cls, datetimeformat): cls.__datetimeformat = datetimeformat


class BarFeed(BarFeedBase, metaclass=BarFeedMeta):
    fields = ["open", "high", "low", "close", "volume", "adjusted"]

    def __init__(self, feed, *args, starttime, stoptime, frequency, length=None, **kwargs):
        assert isinstance(feed, Feed)
        assert frequency in Frequency.__members__
        super().__init__(frequency, length)
        bars = {}
        for content in feed(starttime=starttime, stoptime=stoptime):
            key = content["ticker"]
            index = content["datetime"]
            values = [index] + [content[field] for field in self.__class__.fields]
            if key not in bars.keys():
                bars[key] = []
            bars[key].append(values)
        for key, values in bars.items():
            self.addBarsFromSequence(key, BasicBar(*values, frequency))

    def price(self, ticker): return self[ticker].getPriceDataSeries()
    def open(self, ticker): return self[ticker].getOpenDataSeries()
    def close(self, ticker): return self[ticker].getOpenDataSeries()
    def high(self, ticker): return self[ticker].getHighDataSeries()
    def low(self, ticker): return self[ticker].getLowDataSeries()
    def volume(self, ticker): return self[ticker].getVolumeDataSeries()
    def adjusted(self, ticker): return self[ticker].getAdjCloseDataSeries()

    def barsHaveAdjClose(self):
        return True






