# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Target Objects
@author: Jack Kirby Cook

"""

import xarray as xr
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
Valuation = ntuple("Valuation", "apy tau income cost")
Cashflow = ntuple("Cashflow", "price spot future")


class TargetComparison(object):
    def __eq__(self, other): return [(value - comparable) / value <= 0.01 for value, comparable in zip(self, other)]
    def __ne__(self, other): return not self.__eq__(other)


class TargetValuation(Valuation, TargetComparison): pass
class TargetCashflow(Cashflow, TargetComparison): pass


class TargetSecurity(Security, metaclass=SubclassMeta):
    def __new__(cls, security, *args, **kwargs):
        assert cls.__formatting__ is not None
        if cls is TargetSecurity:
            subcls = cls[str(security.instrument)]
            return subcls(security, *args, **kwargs)
        return super().__new__(security.instrument, security.position)


class TargetStock(TargetSecurity, key="stock"):
    def __init__(self, *args, ticker, **kwargs):
        self.__ticker = ticker

    def __str__(self):
        security = "|".join([str(content.name).lower() for content in self]).title()
        contents = dict(ticker=self.ticker)
        string = "{}[{ticker}]".format(security, **contents)
        return string

    @property
    def ticker(self): return self.__ticker


class TargetOption(TargetSecurity, keys=["put", "call"]):
    def __init__(self, security, *args, expire, **kwargs):
        self.__strike = kwargs[str(security)]
        self.__expire = expire

    def __str__(self):
        security = "|".join([str(content.name).lower() for content in self]).title()
        contents = dict(ticker=self.ticker, strike=self.strike, expire=self.expire)
        string = "{}[{ticker}, ${strike:.2f}, {expire}]".format(security, **contents)
        return string

    @property
    def strike(self): return self.__strike
    @property
    def expire(self): return self.__expire


class TargetStrategy(Strategy):
    def __new__(cls, strategy, *args, **kwargs): return super().__new__(cls, *strategy)
    def __init__(self, *args, securities, valuation, cashflow, **kwargs):
        self.__securities = securities
        self.__valuation = valuation
        self.__cashflow = cashflow

    def __str__(self):
        strategy = "|".join([str(content.name).lower() for content in self]).title()
        contents = {key: value for key, value in zip(Valuation._fields, self.valuation)}
        string = "{}[{apy:.2f}%, ${income:.2f}|${cost:.2f}, ☀{tau:.0f}]".format(strategy, **contents)
        string = "\n".join([string] + [str(security) for security in self.securities])
        return string

    @property
    def securities(self): return self.__securities
    @property
    def valuation(self): return self.__valuation
    @property
    def cashflow(self): return self.__cashflow


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
                valuation = Valuation(*[record[field] for field in TargetValuation._fields])
                cashflow = Cashflow(*[record[field] for field in TargetCashflow._fields])
                strategy = TargetStrategy(strategy, securities=securities, valuation=valuation, cashflow=cashflow, **record)
                yield strategy

    @staticmethod
    def partitions(dataframe):
        if not hasattr(dataframe, "npartitions"):
            yield dataframe
            return
        for index in dataframe.npartitions:
            partition = dataframe.get_partition(index).compute()
            yield partition



