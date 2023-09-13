# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Security Objects
@author: Jack Kirby Cook

"""

import os.path
import numpy as np
import pandas as pd
from enum import IntEnum
from datetime import date as Date
from datetime import datetime as Datetime
from collections import namedtuple as ntuple

from support.pipelines import Processor, Saver, Loader

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DateRange", "Security", "Securities", "Instruments", "Positions", "HistoryLoader", "HistoryCalculator", "HistorySaver", "SecurityLoader", "SecurityCalculator", "SecuritySaver"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


Instruments = IntEnum("Instrument", ["PUT", "CALL", "STOCK"], start=1)
Positions = IntEnum("Position", ["LONG", "SHORT"], start=1)
class Security(ntuple("Security", "instrument position")):
    def __new__(cls, instrument, position, *args, **kwargs): return super().__new__(cls, instrument, position)
    def __init__(self, *args, payoff, **kwargs): self.__payoff = payoff
    def __str__(self): return "|".join([str(value.name).lower() for value in self if bool(value)])

    @property
    def payoff(self): return self.__payoff


class SecuritiesMeta(type):
    def __getitem__(cls, string):
        instrument, position = str(string).split("|")
        instrument, position = str(instrument).title(), str(position).title()
        try:
            option = getattr(cls.Option, str(instrument).title())
            return getattr(option, str(position).title())
        except AttributeError:
            stock = getattr(cls, str(instrument).title())
            return getattr(stock, str(position).title())


class Securities(metaclass=SecuritiesMeta):
    class Stock:
        Long = Security(Instruments.STOCK, Positions.LONG, payoff=lambda x: np.copy(x))
        Short = Security(Instruments.STOCK, Positions.SHORT, payoff=lambda x: -np.copy(x))
    class Option:
        class Put:
            Long = Security(Instruments.PUT, Positions.LONG, payoff=lambda x, k: np.maximum(k - x, 0))
            Short = Security(Instruments.PUT, Positions.SHORT, payoff=lambda x, k: - np.maximum(k - x, 0))
        class Call:
            Long = Security(Instruments.CALL, Positions.LONG, payoff=lambda x, k: np.maximum(x - k, 0))
            Short = Security(Instruments.CALL, Positions.SHORT, payoff=lambda x, k: - np.maximum(x - k, 0))


class DateRange(ntuple("DateRange", "minimum maximum")):
    def __new__(cls, dates):
        assert isinstance(dates, list)
        assert all([isinstance(date, Date) for date in dates])
        dates = [date if isinstance(date, Date) else date.date() for date in dates]
        return super().__new__(cls, min(dates), max(dates)) if dates else None

    def __repr__(self): return "{}({}, {})".format(self.__class__.__name__, repr(self.minimum), repr(self.maximum))
    def __str__(self): return "{}|{}".format(str(self.minimum), str(self.maximum))
    def __bool__(self): return self.minimum < self.maximum
    def __len__(self): return (self.maximum - self.minimum).days
    def __contains__(self, date): return self.minimum <= date <= self.maximum


class HistoryLoader(Loader):
    def execute(self, ticker, *args, **kwargs):
        file = os.path.join(self.repository, str(ticker) + ".csv")
        datatypes = {"ticker": str, "open": np.float32, "close": np.float32, "high": np.float32, "low": np.float32, "price": np.float32, "volume": np.int64}
        history = self.read(file=file, datatypes=datatypes, datetypes=["date"])
        yield ticker, history


class HistoryCalculator(Processor):
    def execute(self, contents, *args, dates=None, **kwargs):
        ticker, history = contents
        assert isinstance(history, pd.DataFrame)
        assert isinstance(dates, (DateRange, type(None)))
        history = history.where(history["date"] in dates) if bool(dates) else history
        yield ticker, history


class HistorySaver(Saver):
    def execute(self, contents, *args, **kwargs):
        ticker, history = contents
        assert isinstance(history, pd.DataFrame)
        file = os.path.join(self.repository, str(ticker) + ".csv")
        self.write(history, file=file, mode="a")


class SecurityLoader(Loader):
    def execute(self, ticker, *args, **kwargs):
        folder = os.path.join(self.repository, str(ticker))
        for foldername in os.listdir(folder):
            expire = Datetime.strptime(os.path.splitext(foldername)[0], "%Y%m%d").date()
            securities = {key: value for key, value in self.securities(ticker, expire)}
            yield ticker, expire, securities

    def securities(self, ticker, expire):
        datatypes = {"ticker": str, "strike": np.float32, "price": np.float32, "size": np.float32, "interest": np.int32, "volume": np.int64}
        folder = os.path.join(self.repository, str(ticker), str(expire.strftime("%Y%m%d")))
        for filename in os.listdir(folder):
            security = Securities[str(filename).split(".")[0]]
            file = os.path.join(folder, filename)
            securities = self.read(file=file, datatypes=datatypes, datetypes=["date", "datetime", "expire"])
            yield security, securities


class SecurityCalculator(Processor):
    def execute(self, contents, *args, **kwargs):
        ticker, expire, securities = contents
        assert isinstance(securities, dict)
        assert all([isinstance(security, pd.DataFrame) for security in securities.values()])
        securities = self.parser(securities, *args, **kwargs)
        return ticker, expire, securities

    @staticmethod
    def parser(securities, *args, size=None, interest=None, volume=None, **kwargs):
        securities = securities.where(securities["size"] >= size) if bool(size) else securities
        securities = securities.where(securities["interest"] >= interest) if bool(interest) else securities
        securities = securities.where(securities["volume"] >= volume) if bool(volume) else securities
        return securities


class SecuritySaver(Saver):
    def execute(self, contents, *args, **kwargs):
        ticker, expire, securities = contents
        assert isinstance(securities, dict)
        assert all([isinstance(security, pd.DataFrame) for security in securities.values()])
        folder = os.path.join(self.repository, str(ticker), str(expire.strftime("%Y%m%d")))
        if not os.path.isdir(folder):
            os.mkdir(folder)
        for security, dataframe in securities.items():
            file = str(security) + ".csv"
            self.write(dataframe, file=file, mode="a")






