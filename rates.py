# -*- coding: utf-8 -*-
"""
Created on Tues Nov 24 2026
@name:   Finance Interest Objects
@author: Jack Kirby Cook
@file:   finance/interest.py

"""

import numpy as np
import pandas_market_calendars as calenders
from enum import Enum
from dataclasses import dataclass
from datetime import date as Date
from datetime import timedelta as Timedelta

from support.dispatchers import ValueDispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Interest", "Dividends"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


nyse = calenders.get_calendar("NYSE")


class Count(Enum):
    TRADING = 252
    CALENDER = 365


@dataclass(frozen=True, slots=True)
class Rate:
    date: Date; value: float

    @ValueDispatcher(locator=1)
    def days(self, today, count): raise ValueError(count)
    def tau(self, today, count): return self.days(today, count) / int(count)

    @days.register(Count.TRADING)
    def _(self, today, count): return len(nyse.valid_days(start_date=today + Timedelta(days=1), end_date=self.date))
    @days.register(Count.CALENDER)
    def _(self, today, count): return int((self.date - today).days)


class Rates:
    rates: list[Rate]; today: Date; count: Count = Count.TRADING

    @property
    def days(self): return np.array([rate.days(self.today, self.count) for rate in self.rates], dtype=np.int32)
    @property
    def tau(self): return np.array([rate.tau(self.today, self.count) for rate in self.rates], dtype=np.float32)
    @property
    def rate(self): return np.array([rate.value for rate in self.rates], dtype=np.float32)


class Interest(Rates):
    pass


class Dividends(Rates):
    pass



