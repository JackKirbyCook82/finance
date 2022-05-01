# -*- coding: utf-8 -*-
"""
Created on Weds Apr 27 2022
@name:   Trading Options
@author: Jack Kirby Cook

"""


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
    pass


class Option(ABC, metaclass=OptionMeta):
    # Continuous Interest Rate
    # Continuous Dividend Rate

    def __init__(self, k, tk, r, q):
        assert isinstance(tk, (Date, Datetime))
        self.__k = k
        self.__tk = tk
        self.__r = r
        self.__q = q

    def __call__(self, si, ti, vi):
        return self.value(si, ti, vi)

    @property
    def k(self): return self.__k
    @property
    def tk(self): return self.__tk
    @property
    def r(self): return self.__r
    @property
    def q(self): return self.__q

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

    def tau(self, ti):
        assert isinstance(ti, (Date, Datetime))
        assert ti < self.tk
        return (self.tk.date() - ti.date()).days / 365.0

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
        return (rpv * pdf) / (si * vi * math.sqft(t))

    @staticmethod
    def cdf(x): return norm.cdf(x)
    @staticmethod
    def pdf(x): return math.pow(math.sqrt(2 * math.pi), -1) * math.exp(math.power(x, 2) / 2)

    def pv(self, ti, rate):
        return math.exp(-rate * self.tau(ti))

    def dnv(self, ti, si, vi):
        t = self.tau(ti)
        w = math.pow(vi, 2) / 2
        a = math.log(si / self.k)
        b = t * (self.r - self.q + w)
        c = vi * math.pow(t, 2)
        d1 = (a + b) / c
        d2 = d1 - c
        return d1, d2


class Call(Option, key=OptionType.CALL):
    def value(self, ti, si, vi):
        rpv = self.pv(ti, self.r)
        qpv = self.pv(ti, self.q)
        d1, d2 = self.dnv(ti, si, vi)
        n1, n2 = self.cdf(d1), self.cdf(d2)
        return (si * qpv * n1) - (self.k * rpv * n2)

    def delta(self, ti, si, vi):
        qpv = self.pv(ti, self.q)
        d1, _ = self.dnv(ti, si, vi)
        return qpv * self.pdf(d1)

    def kappa(self, ti, si, vi):
        rpv = self.pv(ti, self.r)
        _, d2 = self.dnv(ti, si, vi)
        return rpv * self.pdf(d2)

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
    def value(self, ti, si, vi):
        rpv = self.pv(ti, self.r)
        qpv = self.pv(ti, self.q)
        d1, d2 = self.dnv(ti, si, vi)
        n1, n2 = self.cdf(-d1), self.cdf(-d2)
        return (self.k * rpv * n2) - (si * qpv * n1)

    def delta(self, ti, si, vi):
        qpv = self.pv(ti, self.q)
        d1, _ = self.dnv(ti, si, vi)
        return qpv * (self.pdf(d1) - 1)

    def kappa(self, ti, si, vi):
        rpv = self.pv(ti, self.r)
        _, d2 = self.dnv(ti, si, vi)
        return rpv * (self.pdf(d2) - 1)

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


