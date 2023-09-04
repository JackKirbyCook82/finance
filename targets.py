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


class TargetSecurity(ntuple("Security", "instrument position ticker"), ABC, metaclass=SubclassMeta):
    def __new__(cls, security, *args, ticker, **kwargs):
        if cls is TargetSecurity:
            subcls = cls[str(security.instrument)]
            return subcls(security, *args, ticker=ticker, **kwargs)
        return super().__new__(security.instrument, security.position, ticker)

    def __init__(self, security, *args, **kwargs): self.__payoff = security.payoff
    def __call__(self, domain, *args, **kwargs): return self.execute(domain, *args, **kwargs)

    @abstractmethod
    def execute(self, domain, *args, **kwargs): pass
    @property
    def payoff(self): return self.__payoff


class TargetStock(TargetSecurity, key="stock"):
    def execute(self, domain, *args, **kwargs):
        return self.payoff(domain)


class TargetOption(TargetSecurity, keys=["put", "call"]):
    def __init__(self, security, *args, expire, **kwargs):
        self.__strike = kwargs[str(security)]
        self.__expire = expire

    def execute(self, domain, *args, **kwargs):
        return self.payoff(domain, self.strike)

    @property
    def strike(self): return self.__strike
    @property
    def expire(self): return self.__expire


class TargetValuation(ntuple("Valuation", "price value cost apy tau")):
    def __new__(cls, *args, **kwargs):
        values = [kwargs.get(field, None) for field in cls._fields]
        return super().__new__(cls, *values)


class TargetStrategy(ntuple("Strategy", "spread security instrument")):
    def __new__(cls, strategy, securities, valuation):
        return super().__new__(*strategy)

    def __init__(self, strategy, securities, valuation):
        self.__securities = securities
        self.__valuation = valuation

    def __call__(self, domain, *args, **kwargs):
        return self.execute(domain, *args, **kwargs)

    def execute(self, domain, *args, **kwargs):
        payoffs = np.array([security(domain, *args, **kwargs) for security in self.securities])
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



