# -*- coding: utf-8 -*-
"""
Created on Weds Apr 27 2022
@name:   Trading Feed Objects
@author: Jack Kirby Cook

"""

import warnings
import logging
from abc import ABCMeta
from datetime import date as Date
from datetime import datetime as Datetime
from collections import OrderedDict as ODict
from collections import namedtuple as ntuple

import pandas as pd
from pyalgotrade.bar import BasicBar, Frequency
from pyalgotrade.barfeed.membf import BarFeed as BarFeedBase


from utilities.files import ZIPCSVFile

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["BarReader", "StrategyFeed", "HistoryFeed", "Frequency"]
__copyright__ = "Copyright 2022, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class BarReader(object):
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


class BarFeedMeta(ABCMeta):
    fields = ["open", "close", "high", "low", "volume", "adjusted"]

    def __call__(cls, reader, *args, ticker=None, starttime=None, stoptime=None, **kwargs):
        assert isinstance(reader, BarReader)
        assert isinstance(starttime, Datetime) and isinstance(stoptime, Date)
        feed = cls.feed(reader, ticker, starttime, stoptime)
        instance = super(BarFeedMeta, cls).__call__(*args, feed=feed, **kwargs)
        return instance

    def feed(cls, reader, ticker, starttime, stoptime):
        for content in reader(ticker=ticker, starttime=starttime, stoptime=stoptime):
            ticker = content["ticker"]
            index = content["index"]
            contents = ODict([(field, content[field]) for field in cls.fields])
            yield ticker, index, contents


class BarFeedProxy(ntuple("BarFeedProxy", ["open", "close", "high", "low", "volume", "adjusted"])):
    pass


class StrategyFeed(BarFeedBase, metaclass=BarFeedMeta):
    def __init__(self, *args, feed, frequency, length=None, **kwargs):
        assert frequency in Frequency.__members__
        super().__init__(frequency, length)
        for ticker, index, contents in iter(feed):
            bar = BasicBar([index] + list(contents.values()), frequency)
            self.addBarsFromSequence(ticker, bar)

    def __getitem__(self, ticker):
        attrs = ["getOpenDataSeries", "getCloseDataSeries", "getHighDataSeries", "getLowDataSeries", "getVolumeDataSeries", "getAdjCloseDataSeries"]
        return BarFeedProxy(*[getattr(super().__getitem__(ticker), attr) for attr in attrs])

    @staticmethod
    def barsHaveAdjClose():
        return True


class HistoryFeed(dict, metaclass=BarFeedMeta):
    def __init__(self, *args, feed, **kwargs):
        records = [{"ticker": ticker, "index": index, **contents} for ticker, index, contents in iter(feed)]
        groups = pd.DataFrame(records).set_index("index", drop=True, inplace=False).groupby("ticker")
        super().__init__({ticker: dataframe for ticker, dataframe in iter(groups)})

    def __getitem__(self, ticker):
        columns = ["open", "close", "high", "low", "volume", "adjusted"]
        return BarFeedProxy(*[lambda: super().__getitem__(ticker)[column] for column in columns])

    @staticmethod
    def barsHaveAdjClose():
        return True



