# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Target Objects
@author: Jack Kirby Cook

"""

import numpy as np
import xarray as xr
from abc import ABC, abstractmethod
from collections import namedtuple as ntuple

from support.meta import SubclassMeta
from support.pipelines import Processor

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TargetCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


Security = ntuple("Security", "instrument position")
Strategy = ntuple("Strategy", "spread security instrument")
Valuation = ntuple("Valuation", "price value cost apy tau")
class Target(ABC): pass


class TargetSecurity(Security, Target, metaclass=SubclassMeta):
    def __new__(cls, security, *args, **kwargs):
        if cls is TargetSecurity:
            subcls = cls[str(security.instrument)]
            return subcls(security, *args, **kwargs)
        return super().__new__(security.instrument, security.position)

    def __init__(self, security, *args, ticker, **kwargs):
        self.__function = security.payoff
        self.__ticker = ticker

    @abstractmethod
    def payoff(self, domain, *args, **kwargs): pass
    @abstractmethod
    def todict(self): pass

    @property
    def ticker(self): return self.__ticker
    @property
    def function(self): return self.__function


class TargetStock(TargetSecurity, key="stock"):
    def todict(self): return dict(ticker=self.ticker)
    def payoff(self, domain, *args, **kwargs):
        return self.function(domain)


class TargetOption(TargetSecurity, keys=["put", "call"]):
    def __init__(self, security, *args, expire, **kwargs):
        super().__init__(*args, **kwargs)
        self.__strike = kwargs[str(security)]
        self.__expire = expire

    def todict(self): return dict(ticker=self.ticker, strike=self.strike, expire=self.expire)
    def payoff(self, domain, *args, **kwargs):
        return self.function(domain, self.strike)

    @property
    def strike(self): return self.__strike
    @property
    def expire(self): return self.__expire


class TargetValuation(Valuation, Target):
    def __new__(cls, *args, **kwargs):
        values = [kwargs.get(field, None) for field in cls._fields]
        return super().__new__(cls, *values)


class TargetStrategy(Strategy, Target):
    def __new__(cls, strategy, securities, valuation):
        return super().__new__(*strategy)

    def __init__(self, strategy, securities, valuation):
        self.__securities = securities
        self.__valuation = valuation

    def payoff(self, domain, *args, **kwargs):
        payoffs = np.array([security.payoff(domain, *args, **kwargs) for security in self.securities])
        return np.sum(payoffs, axis=0)

    @property
    def securities(self): return self.__securities
    @property
    def valuation(self): return self.__valuation


class TargetCalculator(Processor):
    def execute(self, contents, *args, apy, **kwargs):
        ticker, expire, strategy, valuations = contents
        assert isinstance(valuations, xr.Dataset)
        dataframe = valuations.to_dask_dataframe() if bool(valuations.chunks) else valuations.to_dataframe()
        dataframe = dataframe.where(dataframe["apy"] >= apy)
        dataframe = dataframe.dropna(how="all")
        for partition in self.partitions(dataframe):
            partition = partition.sort_values("apy", axis=1, ascending=False, ignore_index=True, inplace=False)
            for record in partition.to_dict("records"):
                securities = [TargetSecurity(security, **record) for security in strategy.securities]
                valuation = TargetValuation(**record)
                strategy = TargetStrategy(strategy, securities, valuation)
                yield strategy

    @staticmethod
    def partitions(dataframe):
        if not hasattr(dataframe, "npartitions"):
            yield dataframe
            return
        for index in dataframe.npartitions:
            partition = dataframe.get_partition(index).compute()
            yield partition



