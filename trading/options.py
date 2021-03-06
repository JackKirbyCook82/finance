# -*- coding: utf-8 -*-
"""
Created on Weds Apr 27 2022
@name:   Trading Options Objects
@author: Jack Kirby Cook

"""

import warnings
import logging
import math
import calendar
import numpy as np
from enum import Enum
from scipy.stats import norm
from datetime import datetime as Datetime
from datetime import timedelta as Timedelta
from dateutil.rrule import rrule, MONTHLY
from abc import ABC, ABCMeta, abstractmethod

from utilities.meta import RegistryMeta
from utilities.parsers import dateparser

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["OptionType", "Option", "Call", "Put"]
__copyright__ = "Copyright 2022, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


continuous_rate = lambda apy: math.log(apy + 1)
daily_volatility = lambda x: np.std(np.diff(x) / x[1:] * 100)
yearly_volatility = lambda v: v * math.sqrt(252)
first_day = lambda yr, mo: Datetime(yr, mo, 1)
last_day = lambda yr, mo: Datetime(yr, mo, calendar.monthrange(yr, mo)[1])
day_generator = lambda to, tf: (to + Timedelta(to + i) for i in range((tf - to).days))
total_weekdays = lambda to, tf: sum([1 for day in day_generator(to, tf) if day.weekday() < 5])
total_weekends = lambda to, tf: sum([1 for day in day_generator(to, tf) if day.weekday() > 4])
third_friday = lambda yr, mo: [day for day in day_generator(first_day(yr, mo), last_day(yr, mo)) if day.weekday() == 4][2]
expire_dates = lambda to, tf: [third_friday(t.year, t.month) for t in rrule(MONTHLY, dtstart=to, until=tf) if to <= t <= tf]
expire_periods = lambda tx: np.cumsum(np.array([total_weekdays(ti, tj) for ti, tj in zip(tx[:-1], tx[1:])]))


OptionType = Enum("OptionType", "CALL PUT", start=0)


def stock_ratio_function(v, r, q):
    a = lambda zxy, nxy: nxy * (math.pow(v, 2) / 2)
    b = lambda zxy, nxy: nxy * (r - q)
    c = lambda zxy, nxy: np.sqrt(nxy) * zxy * v
    f = lambda zxy, nxy: np.exp(a(nxy, zxy) + b(nxy, zxy) + c(nxy, zxy))
    return f


def create_market(sx, to, tf, r, q, p=[0.1, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 99.9]):
    px = np.array(list(p))
    v = daily_volatility(sx)
    v = yearly_volatility(v)
    f = stock_ratio_function(v, r, q)
    tx = expire_dates(to, tf)
    nx = expire_periods([to, *tx])
    zx = norm.ppf(px)
    zxy, nxy = np.meshgrid(zx, nx)
    kxy = f(zxy, nxy) * sx[-1]
    for t, kx in zip(tx, kxy):
        for k in list(kx):
            yield Option[OptionType.CALL](t, k), Option[OptionType.PUT](t, k)


class OptionMeta(RegistryMeta, ABCMeta):
    def __init__(cls, *args, **kwargs):
        cls.__type = kwargs["type"]
        super(OptionMeta, cls).__init__(*args, **kwargs)

    def __str__(cls): return str(cls.__type)
    def __int__(cls): return int(cls.__type)


class Option(ABC, metaclass=OptionMeta):
    def __init__(self, ticker, tk, k, *args, **kwargs):
        self.__ticker = str(ticker).upper()
        self.__tk = tk
        self.__k = k

    def __str__(self):
        k = "{:08.3f}".format(self.k).replace(".", "")
        tk = self.__dateparser(self.tk)
        yr, mo, day = str(tk.year)[-2:], str(tk.month).rjust(2, "0"), str(tk.day).rjust(2, "0")
        return "".join([self.ticker, yr, mo, day, str(self.__class__)[0], k])

    @property
    def ticker(self): return self.__ticker
    @property
    def tk(self): return self.__tk
    @property
    def k(self): return self.__k

    @abstractmethod
    def intrinsic(self, ti, si, v, r, q): pass
    @abstractmethod
    def value(self, ti, si, v, r, q): pass

    def tau(self, ti, *args):
        ti, tk = dateparser(ti), dateparser(self.tk)
        assert ti <= self.tk
        return total_weekdays(ti, tk) / 252

    @abstractmethod
    def delta(self, ti, si, v, r, q): pass
    @abstractmethod
    def kappa(self, ti, si, v, r, q): pass
    @abstractmethod
    def rho(self, ti, si, v, r, q): pass
    @abstractmethod
    def theta(self, ti, si, v, r, q): pass

    def vega(self, ti, si, v, r, q):
        t = self.tau(ti)
        d1, _ = self.dnv(ti, si, v, r, q)
        pdf = self.pdf(d1)
        qpv = self.pv(ti, q)
        return (si * qpv * math.sqrt(t) * pdf) / 100.0

    def gamma(self, ti, si, v, r, q):
        t = self.tau(ti)
        d1, _ = self.dnv(ti, si, v, r, q)
        pdf = self.pdf(d1)
        qpv = self.pv(ti, q)
        return (qpv * pdf) / (si * v * math.sqrt(t))

    def zeta(self, ti, si, v, r, q):
        t = self.tau(ti)
        _, d2 = self.dnv(ti, si, v, r, q)
        pdf = self.pdf(d2)
        rpv = self.pv(ti, r)
        return (rpv * pdf) / (si * v * math.sqrt(t))

    @staticmethod
    def cdf(x): return norm.cdf(x)
    @staticmethod
    def pdf(x): return math.pow(math.sqrt(2 * math.pi), -1) * math.exp(math.pow(x, 2) / 2)

    def pv(self, ti, rate):
        rate = continuous_rate(rate)
        return math.exp(-rate * self.tau(ti))

    def dnv(self, ti, si, v, r, q):
        t = self.tau(ti)
        r = continuous_rate(r)
        q = continuous_rate(q)
        w = math.pow(v, 2) / 2
        a = math.log(si / self.k)
        b = t * (r - q + w)
        c = v * math.sqrt(t)
        d1 = (a + b) / c
        d2 = d1 - c
        return d1, d2


class Call(Option, type=OptionType.CALL):
    def intrinsic(self, ti, si, v, r, q):
        rpv = self.pv(ti, r)
        fv = max(0, self.k - si)
        return rpv * fv

    def value(self, ti, si, v, r, q):
        rpv = self.pv(ti, r)
        qpv = self.pv(ti, q)
        d1, d2 = self.dnv(ti, si, v, r, q)
        n1, n2 = self.cdf(d1), self.cdf(d2)
        return (si * qpv * n1) - (self.k * rpv * n2)

    def delta(self, ti, si, v, r, q):
        qpv = self.pv(ti, q)
        d1, _ = self.dnv(ti, si, v, r, q)
        return qpv * self.cdf(d1)

    def kappa(self, ti, si, v, r, q):
        rpv = self.pv(ti, r)
        _, d2 = self.dnv(ti, si, v, r, q)
        return rpv * self.cdf(d2)

    def rho(self, ti, si, v, r, q):
        t = self.tau(ti)
        rpv = self.pv(ti, r)
        _, d2 = self.dnv(ti, si, v, r, q)
        n2 = self.cdf(d2)
        return (self.k * t * rpv * n2) / 100.0

    def theta(self, ti, si, v, r, q):
        t = self.tau(ti)
        rpv = self.pv(ti, r)
        qpv = self.pv(ti, q)
        d1, d2 = self.dnv(ti, si, v, r, q)
        n1, n2 = self.cdf(d1), self.cdf(d2)
        pdf = self.pdf(d1)
        a = (si * v * qpv * pdf) / (2 * math.sqrt(t))
        b = r * self.k * rpv * n2
        c = q * si * qpv * n1
        return (-a - b + c) / 252


class Put(Option, type=OptionType.PUT):
    def intrinsic(self, ti, si, v, r, q):
        rpv = self.pv(ti, r)
        fv = max(0, si - self.k)
        return rpv * fv

    def value(self, ti, si, v, r, q):
        rpv = self.pv(ti, r)
        qpv = self.pv(ti, q)
        d1, d2 = self.dnv(ti, si, v, r, q)
        n1, n2 = self.cdf(-d1), self.cdf(-d2)
        return (self.k * rpv * n2) - (si * qpv * n1)

    def delta(self, ti, si, v, r, q):
        qpv = self.pv(ti, self.q)
        d1, _ = self.dnv(ti, si, v, r, q)
        return qpv * (self.cdf(d1) - 1)

    def kappa(self, ti, si, v, r, q):
        rpv = self.pv(ti, r)
        _, d2 = self.dnv(ti, si, v, r, q)
        return rpv * (self.cdf(d2) - 1)

    def rho(self, ti, si, v, r, q):
        t = self.tau(ti)
        rpv = self.pv(ti, r)
        _, d2 = self.dnv(ti, si, v, r, q)
        n2 = self.cdf(-d2)
        return -(self.k * t * rpv * n2) / 100.0

    def theta(self, ti, si, v, r, q):
        t = self.tau(ti)
        rpv = self.pv(ti, r)
        qpv = self.pv(ti, q)
        d1, d2 = self.dnv(ti, si, v, r, q)
        n1, n2 = self.cdf(-d1), self.cdf(-d2)
        pdf = self.pdf(d1)
        a = (si * v * qpv * pdf) / (2 * math.sqrt(t))
        b = r * self.k * rpv * n2
        c = q * si * qpv * n1
        return (-a + b - c) / 252





