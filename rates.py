# -*- coding: utf-8 -*-
"""
Created on Tues Nov 24 2026
@name:   Finance Interest Objects
@author: Jack Kirby Cook
@file:   finance/interest.py

"""

import numpy as np
import pandas_market_calendars as calenders
from enum import IntEnum
from dataclasses import dataclass
from datetime import date as Date
from datetime import timedelta as Timedelta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Interest", "Dividends"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


nyse = calenders.get_calendar("NYSE")


class Count(IntEnum):
    TRADING = 252
    CALENDER = 365


@dataclass(frozen=True, slots=True)
class Rate:
    today: Date; date: Date; value: float; count: Count = Count.TRADING

    @property
    def tau(self): return int(self.days) / int(self.count)
    @property
    def days(self):
        if self.count == Count.TRADING: return len(nyse.valid_days(start_date=self.today + Timedelta(days=1), end_date=self.date))
        elif self.count == Count.CALENDER: return int((self.date - self.today).days)
        else: raise ValueError(self.count)


class Rates(list[Rate]):
    @property
    def days(self): return np.array([rate.days for rate in self], dtype=np.int32)
    @property
    def tau(self): return np.array([rate.tau for rate in self], dtype=np.float32)
    @property
    def values(self): return np.array(self, dtype=np.float32)


class Interest(Rates):
    pass


class Dividends(Rates):
    pass



