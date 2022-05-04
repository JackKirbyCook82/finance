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
from enum import Enum
from scipy.stats import norm
from datetime import datetime as Datetime
from datetime import date as Date
from datetime import timedelta as Timedelta
from abc import ABC, ABCMeta, abstractmethod

from utilities.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["OptionType", "Option", "Call", "Put"]
__copyright__ = "Copyright 2022, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


continuous_rate = lambda apy: math.log(apy + 1)
first_day = lambda yr, mo: Datetime(yr, mo, 1)
last_day = lambda yr, mo: Datetime(yr, mo, calendar.monthrange(yr, mo)[1])
day_generator = lambda to, tf: (to + Timedelta(to + i) for i in range((tf - to).days))
total_weekdays = lambda to, tf: sum([1 for day in day_generator(to, tf) if day.weekday() < 5])
total_weekends = lambda to, tf: sum([1 for day in day_generator(to, tf) if day.weekday() > 4])
third_friday = lambda yr, mo: [day for day in day_generator(first_day(yr, mo), last_day(yr, mo)) if day.weekday() == 4][2]


def getmarket():
    pass


def createmarket():
    pass


class OptionType(Enum):
    CALL = 0
    PUT = 1


class OptionMeta(RegistryMeta, ABCMeta):
    def __init__(cls, *args, **kwargs):
        cls.__dateformat = kwargs.get("dateformat", getattr(cls, "dateformat", "%Y/%m/%d"))
        cls.__datetimeformat = kwargs.get("datetimeformat", getattr(cls, "datetimeformat", "%Y/%m/%d %H:%M:%S"))

    def __call__(cls, tk, k, r, q):
        dateparser = {str: lambda x: Datetime.strptime(x, cls.dateformat).date(), Date: lambda x: x, Datetime: lambda x: x.date()}
        instance = super(OptionMeta, cls).__call__(tk, k, continuous_rate(r), continuous_rate(q), dateparser=dateparser)
        return instance

    @property
    def dateformat(cls): return cls.__dateformat
    @property
    def datetimeformat(cls): return cls.__datetimeformat
    @dateformat.setter
    def dateformat(cls, dateformat): cls.__dateformat = dateformat
    @datetimeformat.setter
    def datetimeformat(cls, datetimeformat): cls.__datetimeformat = datetimeformat


class Option(ABC, metaclass=OptionMeta):
    def __init__(self, tk, k, r, q, *args, dateparser, tradingdays=252, **kwargs):
        self.__tradingdays = tradingdays
        self.__dateparser = dateparser
        self.__tk = tk
        self.__k = k
        self.__r = r
        self.__q = q

    @property
    def tk(self): return self.__tk
    @property
    def k(self): return self.__k
    @property
    def r(self): return self.__r
    @property
    def q(self): return self.__q

    @abstractmethod
    def intrinsic(self, ti, si, vi): pass
    @abstractmethod
    def value(self, ti, si, vi): pass

    def tau(self, ti, *args):
        ti, tk = self.__dateparser(ti), self.__dateparser(self.tk)
        assert ti <= self.tk
        return total_weekdays(ti, tk) / self.__tradingdays

    @abstractmethod
    def delta(self, ti, si, vi): pass
    @abstractmethod
    def kappa(self, ti, si, vi): pass
    @abstractmethod
    def rho(self, ti, si, vi): pass
    @abstractmethod
    def theta(self, ti, si, vi): pass

    def vega(self, ti, si, vi):
        t = self.tau(ti)
        d1, _ = self.dnv(ti, si, vi)
        pdf = self.pdf(d1)
        qpv = self.pv(ti, self.q)
        return (si * qpv * math.sqrt(t) * pdf) / 100.0

    def gamma(self, ti, si, vi):
        t = self.tau(ti)
        d1, _ = self.dnv(ti, si, vi)
        pdf = self.pdf(d1)
        qpv = self.pv(ti, self.q)
        return (qpv * pdf) / (si * vi * math.sqrt(t))

    def zeta(self, ti, si, vi):
        t = self.tau(ti)
        _, d2 = self.dnv(ti, si, vi)
        pdf = self.pdf(d2)
        rpv = self.pv(ti, self.r)
        return (rpv * pdf) / (si * vi * math.sqrt(t))

    @staticmethod
    def cdf(x): return norm.cdf(x)
    @staticmethod
    def pdf(x): return math.pow(math.sqrt(2 * math.pi), -1) * math.exp(math.pow(x, 2) / 2)

    def pv(self, ti, rate):
        return math.exp(-rate * self.tau(ti))

    def dnv(self, ti, si, vi):
        t = self.tau(ti)
        w = math.pow(vi, 2) / 2
        a = math.log(si / self.k)
        b = t * (self.r - self.q + w)
        c = vi * math.sqrt(t)
        d1 = (a + b) / c
        d2 = d1 - c
        return d1, d2


class Call(Option, key=OptionType.CALL):
    def intrinsic(self, ti, si, vi):
        rpv = self.pv(ti, self.r)
        fv = max(0, self.k - si)
        return rpv * fv

    def value(self, ti, si, vi):
        rpv = self.pv(ti, self.r)
        qpv = self.pv(ti, self.q)
        d1, d2 = self.dnv(ti, si, vi)
        n1, n2 = self.cdf(d1), self.cdf(d2)
        return (si * qpv * n1) - (self.k * rpv * n2)

    def delta(self, ti, si, vi):
        qpv = self.pv(ti, self.q)
        d1, _ = self.dnv(ti, si, vi)
        return qpv * self.cdf(d1)

    def kappa(self, ti, si, vi):
        rpv = self.pv(ti, self.r)
        _, d2 = self.dnv(ti, si, vi)
        return rpv * self.cdf(d2)

    def rho(self, ti, si, vi):
        t = self.tau(ti)
        rpv = self.pv(ti, self.r)
        _, d2 = self.dnv(ti, si, vi)
        n2 = self.cdf(d2)
        return (self.k * t * rpv * n2) / 100.0

    def theta(self, ti, si, vi):
        t = self.tau(ti)
        rpv = self.pv(ti, self.r)
        qpv = self.pv(ti, self.q)
        d1, d2 = self.dnv(ti, si, vi)
        n1, n2 = self.cdf(d1), self.cdf(d2)
        pdf = self.pdf(d1)
        a = (si * vi * qpv * pdf) / (2 * math.sqrt(t))
        b = self.r * self.k * rpv * n2
        c = self.q * si * qpv * n1
        return (-a - b + c) / self.__tradingdays


class Put(Option, key=OptionType.PUT):
    def intrinsic(self, ti, si, vi):
        rpv = self.pv(ti, self.r)
        fv = max(0, si - self.k)
        return rpv * fv

    def value(self, ti, si, vi):
        rpv = self.pv(ti, self.r)
        qpv = self.pv(ti, self.q)
        d1, d2 = self.dnv(ti, si, vi)
        n1, n2 = self.cdf(-d1), self.cdf(-d2)
        return (self.k * rpv * n2) - (si * qpv * n1)

    def delta(self, ti, si, vi):
        qpv = self.pv(ti, self.q)
        d1, _ = self.dnv(ti, si, vi)
        return qpv * (self.cdf(d1) - 1)

    def kappa(self, ti, si, vi):
        rpv = self.pv(ti, self.r)
        _, d2 = self.dnv(ti, si, vi)
        return rpv * (self.cdf(d2) - 1)

    def rho(self, ti, si, vi):
        t = self.tau(ti)
        rpv = self.pv(ti, self.r)
        _, d2 = self.dnv(ti, si, vi)
        n2 = self.cdf(-d2)
        return -(self.k * t * rpv * n2) / 100.0

    def theta(self, ti, si, vi):
        t = self.tau(ti)
        rpv = self.pv(ti, self.r)
        qpv = self.pv(ti, self.q)
        d1, d2 = self.dnv(ti, si, vi)
        n1, n2 = self.cdf(-d1), self.cdf(-d2)
        pdf = self.pdf(d1)
        a = (si * vi * qpv * pdf) / (2 * math.sqrt(t))
        b = self.r * self.k * rpv * n2
        c = self.q * si * qpv * n1
        return (-a + b - c) / self.__tradingdays





