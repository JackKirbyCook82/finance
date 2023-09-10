# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Target Objects
@author: Jack Kirby Cook

"""

import numpy as np
import xarray as xr
from abc import ABC, abstractmethod
from functools import total_ordering
from collections import namedtuple as ntuple
from datetime import datetime as Datetime

from support.meta import SubclassMeta
from support.pipelines import Processor

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TargetCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


@total_ordering
class TargetValuation(ntuple("Valuation", "deadline spot future tau")):
    def __new__(cls, *args, **kwargs):
        values = args[:len(cls._fields)] if len(args) >= len(cls._fields) else [kwargs[field] for field in cls._fields]
        return super().__new__(cls, *values)

    def __bool__(self): return Datetime.now() >= self.deadline if bool(self.deadline) else True
    def __eq__(self, other): return self.apy == other.apy
    def __lt__(self, other): return self.apy < other.apy

    def __str__(self):
        contents = dict(apy=self.apy, tau=self.tau, income=self.income, cost=self.cost)
        string = "[{apy:.2f}%, ${income:.2f}|${cost:.2f}, ☀{tau:.0f}]".format(**contents)
        return string

    @property
    def apy(self): return np.power(self.gains + 1, np.power(self.tau / 365, -1)) - 1
    @property
    def income(self): return + np.maximum(self.spot, 0) + np.maximum(self.future, 0)
    @property
    def cost(self): return - np.minimum(self.spot, 0) - np.minimum(self.future, 0)
    @property
    def gains(self): return self.profit / self.cost
    @property
    def profit(self): return self.income - self.cost


class TargetSecurity(ntuple("Security", "instrument position"), ABC, metaclass=SubclassMeta):
    def __new__(cls, security, *args, **kwargs):
        assert cls.__formatting__ is not None
        if cls is TargetSecurity:
            subcls = cls[str(security.instrument)]
            return subcls(security, *args, **kwargs)
        return super().__new__(security.instrument, security.position)

    def __init__(self, security, *args, **kwargs):
        self.__function = security.payoff

    @abstractmethod
    def payoff(self, domain): pass
    @property
    def function(self): return self.__function


class TargetStock(TargetSecurity, key="stock"):
    def __init__(self, *args, ticker, **kwargs):
        super().__init__(*args, **kwargs)
        self.__ticker = ticker

    def __str__(self):
        security = "|".join([str(content.name).lower() for content in self]).title()
        contents = dict(ticker=self.ticker)
        string = "{}[{ticker}]".format(security, **contents)
        return string

    def payoff(self, domain): return self.function(domain)
    def contents(self): return dict(ticker=self.ticker)

    @property
    def ticker(self): return self.__ticker


class TargetOption(TargetSecurity, keys=["put", "call"]):
    def __init__(self, security, *args, ticker, expire, **kwargs):
        super().__init__(*args, **kwargs)
        self.__strike = kwargs[str(security)]
        self.__ticker = ticker
        self.__expire = expire

    def __str__(self):
        security = "|".join([str(content.name).lower() for content in self]).title()
        contents = dict(ticker=self.ticker, strike=self.strike, expire=self.expire)
        string = "{}[{ticker}, ${strike:.2f}, {expire}]".format(security, **contents)
        return string

    def payoff(self, domain): return self.function(domain, self.strike)
    def contents(self): return dict(ticker=self.ticker, strike=self.strike, expire=self.expire)

    @property
    def ticker(self): return self.__ticker
    @property
    def strike(self): return self.__strike
    @property
    def expire(self): return self.__expire


class TargetStrategy(ntuple("Strategy", "spread security instrument")):
    def __new__(cls, strategy, *args, **kwargs): return super().__new__(cls, *strategy)
    def __init__(self, *args, securities, valuation, **kwargs):
        self.__securities = securities
        self.__valuation = valuation

    def __str__(self):
        strategy = "|".join([str(content.name).lower() for content in self]).title()
        securities = [str(security) for security in self.securities]
        valuation = str(self.valuation)
        return "\n".join([strategy + valuation, *securities])

    @property
    def securities(self): return self.__securities
    @property
    def valuation(self): return self.__valuation

    def ceiling(self, domain): return np.max(self.payoff(domain))
    def floor(self, domain): return np.min(self.payoff(domain))
    def payoff(self, domain):
        payoffs = np.array([security.payoff(domain) for security in self.securities])
        return np.sum(payoffs, axis=0)


class TargetCalculator(Processor):
    def execute(self, contents, *args, apy=None, funds=None, tenure=None, **kwargs):
        ticker, expire, strategy, valuations = contents
        assert isinstance(valuations, xr.Dataset)
        dataframe = valuations.to_dask_dataframe() if bool(valuations.chunks) else valuations.to_dataframe()
        dataframe = dataframe.where(dataframe["apy"] >= float(apy)) if apy is not None else dataframe
        dataframe = dataframe.where(dataframe["cost"] <= float(funds)) if funds is not None else dataframe
        dataframe = dataframe.dropna(how="all")
        for partition in self.partitions(dataframe):
            partition = partition.sort_values("apy", axis=1, ascending=False, ignore_index=True, inplace=False)
            for index, record in enumerate(partition.to_dict("records")):
                deadline = self.deadline(record["time"], tenure)
                valuation = TargetValuation(deadline=deadline, **record)
                if not bool(valuation):
                    continue
                assert self.pctdiff(record["cost"], valuation.cost) <= 0.01
                assert self.pctdiff(record["apy"], valuation.apy) <= 0.01
                securities = [TargetSecurity(security, **record) for security in strategy.securities]
                strategy = TargetStrategy(strategy, valuation=valuation, securities=securities, **record)
                rigorous = self.rigorous(strategy)
                assert self.pctdiff(record["cost"], rigorous.cost) <= 0.01
                assert self.pctdiff(record["apy"], rigorous.apy) <= 0.01
                yield strategy

    @staticmethod
    def deadline(recorded, tenure): return (recorded + tenure) if bool(tenure) else None
    @staticmethod
    def pctdiff(value, other): return (value - other) / other

    @staticmethod
    def rigorous(target):
        valuation = target.valuation
        domain = np.arange(0, 2000, 0.1)
        future = target.payoff(domain)
        return TargetValuation(valuation.time, valuation.spot, future, valuation.tau)

    @staticmethod
    def partitions(dataframe):
        if not hasattr(dataframe, "npartitions"):
            yield dataframe
            return
        for index in dataframe.npartitions:
            partition = dataframe.get_partition(index).compute()
            yield partition



