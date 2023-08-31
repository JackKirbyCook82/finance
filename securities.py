# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Security Objects
@author: Jack Kirby Cook

"""

import pandas as pd
from enum import IntEnum
from datetime import date as Date
from collections import namedtuple as ntuple

from support.pipelines import Processor, Saver, Loader

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DateRange", "Securities", "Instruments", "Positions", "HistoryLoader", "HistoryProcessor", "HistorySaver", "SecurityLoader", "SecurityProcessor", "SecuritySaver"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


Instruments = IntEnum("Security", ["PUT", "CALL", "STOCK"], start=1)
Positions = IntEnum("Position", ["LONG", "SHORT"], start=1)
class Security(ntuple("Security", "instrument position")):
    def __int__(self): return sum([self.instrument * 10, self.position * 1])
    def __str__(self): return "|".join([str(value.name).lower() for value in self if bool(value)])


class Securities:
    class Stock:
        Long = Security(Instruments.STOCK, Positions.LONG)
        Short = Security(Instruments.STOCK, Positions.SHORT)
    class Option:
        class Put:
            Long = Security(Instruments.PUT, Positions.LONG)
            Short = Security(Instruments.PUT, Positions.SHORT)
        class Call:
            Long = Security(Instruments.CALL, Positions.LONG)
            Short = Security(Instruments.CALL, Positions.SHORT)


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
        pass


class HistoryProcessor(Processor):
    def execute(self, contents, *args, dates, **kwargs):
        ticker, history = contents
        assert isinstance(history, pd.DataFrame)


class HistorySaver(Saver):
    def execute(self, contents, *args, **kwargs):
        ticker, history = contents
        assert isinstance(history, pd.DataFrame)


class SecurityLoader(Loader):
    def execute(self, ticker, *args, **kwargs):
        pass


class SecurityProcessor(Processor):
    def execute(self, contents, *args, size, interest, volume, **kwargs):
        ticker, expire, securities = contents
        assert isinstance(securities, dict)
        assert all([isinstance(security, pd.DataFrame) for security in securities.values()])


class SecuritySaver(Saver):
    def execute(self, contents, *args, **kwargs):
        ticker, expire, securities = contents
        assert isinstance(securities, dict)
        assert all([isinstance(security, pd.DataFrame) for security in securities.values()])


