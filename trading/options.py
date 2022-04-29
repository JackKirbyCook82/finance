# -*- coding: utf-8 -*-
"""
Created on Weds Apr 27 2022
@name:   Trading Options
@author: Jack Kirby Cook

"""

import math
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


class OptionMeta(RegistryMeta, ABCMeta):
    pass


class Option(ABC, metaclass=OptionMeta):
    # Continuous Interest Rate
    # Continuous Dividend Rate

    def __init__(self, strike, expire, interest, dividend):
        assert isinstance(expire, (Date, Datetime))
        self.__strike = strike
        self.__expire = expire
        self.__interest = interest
        self.__dividend = dividend

    @property
    def k(self): return self.__strike

    @property
    def r(self): return self.__interest

    @property
    def q(self): return self.__dividend

    @abstractmethod
    def value(self, current, price, volatility): pass
    @abstractmethod
    def delta(self, current, price, volatility): pass
    @abstractmethod
    def rho(self, current, price, volatility): pass
    @abstractmethod
    def theta(self, current, price, volatility): pass

    def gamma(self, i, s, v):
        t = self.tau(i)
        d1, _ = self.dnv(i, s, v)
        pdf = self.pdf(d1)
        qpv = self.pv(i, self.q)
        return (qpv * pdf) / (s * v * math.sqrt(t))

    def vega(self, i, s, v):
        t = self.tau(i)
        d1, _ = self.dnv(i, s, v)
        pdf = self.pdf(d1)
        qpv = self.pv(i, self.q)
        return (s * qpv * math.sqrt(t) * pdf) / 100

    def tau(self, i):
        assert isinstance(i, (Date, Datetime))
        assert i < self.__expire
        return (self.__expire.date() - i.date()).days / 365.0

    def pv(self, i, rate):
        return math.exp(-rate * self.tau(i))

    @staticmethod
    def cdf(x): return norm.cdf(x)
    @staticmethod
    def pdf(x): return math.pow(math.sqrt(2 * math.pi), -1) * math.exp(math.power(x, 2) / 2)

    def dnv(self, i, s, v):
        t = self.tau(i)
        w = math.pow(v, 2) / 2
        a = math.log(s / self.k)
        b = t * (self.r - self.q + w)
        c = v * math.pow(t, 2)
        d1 = (a + b) / c
        d2 = d1 - c
        return d1, d2


class Call(Option, key=OptionType.CALL):
    def value(self, i, s, v):
        rpv = self.pv(i, self.r)
        qpv = self.pv(i, self.q)
        d1, d2 = self.dnv(i, s, v)
        n1, n2 = self.cdf(d1), self.cdf(d2)
        return (s * qpv * n1) - (self.k * rpv * n2)

    def delta(self, i, s, v):
        qpv = self.pv(i, self.q)
        d1, _ = self.dnv(i, s, v)
        return qpv * self.pdf(d1)

    def kappa(self, i, s, v):
        rpv = self.pv(i, self.r)
        _, d2 = self.dnv(i, s, v)
        return rpv * self.pdf(d2)

    def rho(self, i, s, v):
        t = self.tau(i)
        rpv = self.pv(i, self.r)
        _, d2 = self.dnv(i, s, v)
        n2 = self.cdf(d2)
        return (self.k * t * rpv * n2) / 100

    def theta(self, i, s, v):
        t = self.tau(i)
        rpv = self.pv(i, self.r)
        qpv = self.pv(i, self.q)
        d1, d2 = self.dnv(i, s, v)
        n1, n2 = self.cdf(d1), self.cdf(d2)
        pdf = self.pdf(d1)
        a = (s * v * qpv * pdf) / (2 * math.sqrt(t))
        b = self.r * self.k * rpv * n2
        c = self.q * s * qpv * n1
        return (-a - b + c) / 365


class Put(Option, key=OptionType.PUT):
    def value(self, i, s, v):
        rpv = self.pv(i, self.r)
        qpv = self.pv(i, self.q)
        d1, d2 = self.dnv(i, s, v)
        n1, n2 = self.cdf(-d1), self.cdf(-d2)
        return (self.k * rpv * n2) - (s * qpv * n1)

    def delta(self, i, s, v):
        qpv = self.pv(i, self.q)
        d1, _ = self.dnv(i, s, v)
        return qpv * (self.pdf(d1) - 1)

    def kappa(self, i, s, v):
        rpv = self.pv(i, self.r)
        _, d2 = self.dnv(i, s, v)
        return rpv * (self.pdf(d2) - 1)

    def rho(self, i, s, v):
        t = self.tau(i)
        rpv = self.pv(i, self.r)
        _, d2 = self.dnv(i, s, v)
        n2 = self.cdf(-d2)
        return -(self.k * t * rpv * n2) / 100

    def theta(self, i, s, v):
        t = self.tau(i)
        rpv = self.pv(i, self.r)
        qpv = self.pv(i, self.q)
        d1, d2 = self.dnv(i, s, v)
        n1, n2 = self.cdf(-d1), self.cdf(-d2)
        pdf = self.pdf(d1)
        a = (s * v * qpv * pdf) / (2 * math.sqrt(t))
        b = self.r * self.k * rpv * n2
        c = self.q * s * qpv * n1
        return (-a + b - c) / 365


