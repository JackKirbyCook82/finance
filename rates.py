# -*- coding: utf-8 -*-
"""
Created on Tues Nov 24 2026
@name:   Finance Interest Objects
@author: Jack Kirby Cook
@file:   finance/interest.py

"""

import requests
import numpy as np
import pandas as pd
import pandas_market_calendars as calenders
from io import BytesIO
from enum import IntEnum
from dataclasses import dataclass
from datetime import date as Date
from datetime import timedelta as Timedelta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Interest", "Dividend", "Carry"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


nyse = calenders.get_calendar("NYSE")


class CarryError(Exception): pass
class ArbitrageError(Exception): pass
class CurveLimitError(Exception): pass


class BASIS(IntEnum):
    TRADING = 252
    CALENDER = 365

class COMPOUNDING(IntEnum):
    ANNUAL = 1
    SEMIANNUAL = 2
    QUARTERLY = 4
    MONTHLY = 12
    WEEKLY = 52


@dataclass(frozen=True)
class Dividend: date: Date; amount: float


@dataclass(frozen=True)
class Rate:
    tenure: Timedelta
    zero: float
    today: Date

    @property
    def annual(self): return np.exp(self.zero) - 1
    @property
    def discount(self): return 1 / np.exp(self.zero * self.tau)
    @property
    def trading(self): return len(nyse.valid_days(start_date=self.today, end_date=self.maturity))
    @property
    def calender(self): return int(self.tenure.days)
    @property
    def maturity(self): return self.today + self.tenure
    @property
    def tau(self): return self.trading / int(BASIS.TRADING)


@dataclass(frozen=True)
class Interest:
    rates: tuple[Rate, ...]
    today: Date

    def discount(self, date):
        trading = len(nyse.valid_days(start_date=self.today, end_date=date))
        tau = int(trading) / int(BASIS.TRADING)
        taus = np.array(rate.tau for rate in self.rates)
        discounts = np.array(rate.discount for rate in self.rates)
        x = np.r_[0.0, taus]
        y = np.r_[0.0, np.log(discounts)]
        return float(np.exp(np.interp(tau, x, y)))

    @classmethod
    def download(cls, today):
        url = "https://home.treasury.gov/resource-center/" "data-chart-center/interest-rates/pages/xml"
        params = {"data": "daily_treasury_bill_rates", "field_tdr_date_value_month": today.strftime("%Y%m")}
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        raw = pd.read_xml(BytesIO(response.content), xpath=".//m:properties", namespaces={"m": "http://schemas.microsoft.com/" "ado/2007/08/dataservices/metadata"})
        columns = {f"ROUND_B1_CLOSE_{weeks:.0f}_2": int(weeks * 7) for weeks in (4, 6, 8, 13, 17, 26, 52)}
        raw["NEW_DATE"] = pd.to_datetime(raw["NEW_DATE"])
        raw = raw.set_index("NEW_DATE")
        raw = raw[list(columns)].rename(columns=columns)
        raw = raw.apply(pd.to_numeric, errors="coerce") / 100
        data = raw.sort_index().sort_index(axis=1)
        days = data.columns.to_numpy(dtype=float)
        rates = - np.log1p(- data * days / 360) / (days / 365)
        curve = zip(days, rates)
        rates = tuple(Rate(Timedelta(days=days), zero, today) for days, zero in curve)
        return cls(rates, today)


@dataclass(frozen=True)
class Carry:
    dividends: tuple[Dividend]
    interest: Interest
    today: Date

    def forward(self, spot, date):
        function = lambda dividend: dividend.amount * self.interest.discount(dividend.date)
        dividends = sum(function(dividend) for dividend in self.dividends if self.today < dividend.date < date)
        discount = self.interest.discount(date)
        if spot <= dividends: raise ArbitrageError()
        return (spot - dividends) / discount



