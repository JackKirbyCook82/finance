# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Target Objects
@author: Jack Kirby Cook

"""

import xarray as xr
from collections import namedtuple as ntuple

from support.pipelines import Processor

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TargetCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


Stock = ntuple("Stock", "instrument position ticker")
Option = ntuple("Option", "instrument position ticker expire strike")
Strategy = ntuple("Strategy", "spread security position")
Valuation = ntuple("Valuation", "price cost apy tau")


class TargetStrategy(Strategy):
    def __new__(cls, strategy, *args, ticker, expire, **kwargs):
        cls = super().__new__(cls, *strategy)
        options = [Option(*security, ticker, expire, kwargs[str(security)]) for security in strategy.securities if str(security) in kwargs.keys()]
        stocks = [Stock(*security, ticker) for security in strategy.securities if str(security) in kwargs.keys()]
        cls.valuation = Valuation(*[kwargs[field] for field in Valuation._fields])
        cls.securities = options + stocks
        return cls

    def __init__(self, *args, time, **kwargs):
        self.__valuation = None
        self.__securities = []
        self.__time = time

    @property
    def valuation(self): return self.__valuation
    @valuation.setter
    def valuation(self, valuation): self.__valuation = valuation
    @property
    def securities(self): return self.__securities
    @securities.setter
    def securities(self, securities): self.__securities = securities
    @property
    def time(self): return self.__time


class TargetCalculator(Processor):
    def execute(self, contents, *args, apy, funds, tenure, **kwargs):
        ticker, expire, strategy, valuations = contents
        assert isinstance(valuations, xr.Dataset)
        dataframe = valuations.to_dask_dataframe() if bool(valuations.chunks) else valuations.to_dataframe()
        dataframe = dataframe.where(dataframe["apy"] >= apy)
        dataframe = dataframe.where(dataframe["cost"] <= funds)
        dataframe = dataframe.dropna(how="all")
        for partition in self.partitions(dataframe):
            partition = partition.sort_values("apy", axis=1, ascending=False, ignore_index=True, inplace=False)
            for record in partition.to_dict("records"):
                target = TargetStrategy(strategy, time=, **record)
                if target.cost > funds:
                    continue
                if target.duration > tenure:
                    return
                yield target

    @staticmethod
    def partitions(dataframe):
        if not hasattr(dataframe, "npartitions"):
            yield dataframe
            return
        for index in dataframe.npartitions:
            partition = dataframe.get_partition(index).compute()
            yield partition



