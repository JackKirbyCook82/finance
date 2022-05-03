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
from abc import ABC, ABCMeta, abstractmethod

from utilities.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["OptionType", "Option", "Call", "Put"]
__copyright__ = "Copyright 2022, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class OptionType(Enum):
    CALL = 0
    PUT = 1


class Days(Enum):
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


def days(year, month, day):
    assert isinstance(year, int) and isinstance(month, int)
    assert day in Days.__members__
    x = calendar.Calendar(firstweekday=calendar.SUNDAY)(year, month)
    y = [d for w in x for d in w if d.weekday() == day and d.month == month]
    return y


class OptionMeta(RegistryMeta, ABCMeta):
    def __init__(cls, *args, **kwargs):
        cls.__dateformat = kwargs.get("dateformat", getattr(cls, "dateformat", "%Y/%m/%d"))
        cls.__datetimeformat = kwargs.get("datetimeformat", getattr(cls, "datetimeformat", "%Y/%m/%d %H:%M:%S"))

    def __call__(cls, tk, k, r, q):
        r = cls.continuous(r)
        q = cls.continuous(q)
        instance = super(OptionMeta, cls).__call__(tk, k, r, q)
        return instance

    def days(cls, ti, tk):
        ti = cls.parsers[type(ti)](ti)
        tk = cls.parsers[type(tk)](tk)
        return (tk - ti).days

    @staticmethod
    def continuous(apy): return math.log(apy + 1)
    @property
    def parsers(cls): return {str: lambda x: Datetime.strptime(x, cls.dateformat).date(), Date: lambda x: x, Datetime: lambda x: x.date()}
    @property
    def dateformat(cls): return cls.__dateformat
    @property
    def datetimeformat(cls): return cls.__datetimeformat
    @dateformat.setter
    def dateformat(cls, dateformat): cls.__dateformat = dateformat
    @datetimeformat.setter
    def datetimeformat(cls, datetimeformat): cls.__datetimeformat = datetimeformat


class Option(ABC, metaclass=OptionMeta):
    # Continuous Interest Rate
    # Continuous Dividend Rate
    # Annual Volatility

    def __init__(self, tk, k, r, q):
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
    @abstractmethod
    def delta(self, ti, si, vi): pass
    @abstractmethod
    def kappa(self, ti, si, vi): pass
    @abstractmethod
    def rho(self, ti, si, vi): pass
    @abstractmethod
    def theta(self, ti, si, vi): pass

    def tau(self, ti, *args):
        assert ti < self.tk
        return self.__class__.days(ti, self.tk) / 365.0

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

    def greeks(self, ti, si, vi):
        greeks = {"delta": self.delta(ti, si, vi), "kappa": self.kappa(ti, si, vi), "rho": self.rho(ti, si, vi), "theta": self.theta(ti, si, vi), "vega": self.vega(ti, si, vi),
                  "gamma": self.gamma(ti, si, vi), "zeta": self.zeta(ti, si, vi), "tau": self.tau(ti, si, vi)}
        return greeks

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
        return (-a - b + c) / 365.0


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
        return (-a + b - c) / 365.0





