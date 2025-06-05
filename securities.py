# -*- coding: utf-8 -*-
"""
Created on Tues May 13 2025
@name:   Security Objects
@author: Jack Kirby Cook

"""
import numpy as np
import pandas as pd

from finance.variables import Querys, Variables
from support.mixins import Emptying, Sizing, Partition, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["PricingCalculator", "SecurityCalculator"]
__copyright__ = "Copyright 2025, Jack Kirby Cook"
__license__ = "MIT License"


class PricingCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init_subclass__(cls, *args, query, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls.__query__ = query

    def __init__(self, *args, pricing, **kwargs):
        super().__init__(*args, **kwargs)
        self.__pricing = pricing

    def execute(self, securities, *args, **kwargs):
        assert isinstance(securities, pd.DataFrame)
        if self.empty(securities): return
        querys = self.keys(securities, by=self.query)
        querys = ",".join(list(map(str, querys)))
        securities = self.calculate(securities, *args, **kwargs)
        size = self.size(securities)
        self.console(f"{str(querys)}[{int(size):.0f}]")
        if self.empty(securities): return
        yield securities

    def calculate(self, securities, *args, **kwargs):
        assert isinstance(securities, pd.DataFrame)
        pricing = securities.apply(self.pricing, axis=1).rename("price")
        securities = pd.concat([securities, pricing], axis=1)
        return securities

    @property
    def query(self): return type(self).__query__
    @property
    def pricing(self): return self.__pricing


class AnalyticCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init_subclass__(cls, *args, query, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls.__query__ = query

    def execute(self, securities, technicals, *args, **kwargs):
        assert isinstance(securities, pd.DataFrame) and isinstance(technicals, pd.DataFrame)
        if self.empty(securities): return
        querys = self.keys(securities, by=self.query)
        querys = ",".join(list(map(str, querys)))
        securities = self.calculate(securities, technicals, *args, **kwargs)
        size = self.size(securities)
        self.console(f"{str(querys)}[{int(size):.0f}]")
        if self.empty(securities): return
        yield securities

    @staticmethod
    def calculate(securities, technicals, *args, **kwargs):
        assert isinstance(securities, pd.DataFrame) and isinstance(technicals, pd.DataFrame)
        mask = technicals["date"] == technicals["date"].max()
        technicals = technicals.where(mask).dropna(how="all", inplace=False)
        technicals = technicals.drop(columns="date", inplace=False)
        securities = securities.merge(technicals, how="left", on=list(Querys.Symbol), sort=False, suffixes=("", ""))
        return securities

    @property
    def query(self): return type(self).__query__
    @property
    def pricing(self): return self.__pricing


class SecurityCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def execute(self, stocks, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if self.empty(options): return
        settlements = self.keys(options, by=Querys.Settlement)
        settlements = ",".join(list(map(str, settlements)))
        securities = self.calculate(stocks, options, *args, **kwargs)
        size = self.size(securities)
        self.console(f"{str(settlements)}[{int(size):.0f}]")
        if self.empty(securities): return
        yield securities

    def calculate(self, stocks, options, *args, **kwargs):
        assert isinstance(stocks, pd.DataFrame) and isinstance(options, pd.DataFrame)
        underlying = stocks.apply(lambda series: series["price"], axis=1).rename("underlying")
        stocks = pd.concat([stocks[list(Querys.Symbol)], underlying], axis=1)
        options = options.merge(stocks, how="left", on=list(Querys.Symbol), sort=False, suffixes=("", ""))
        options = list(self.calculator(options, *args, **kwargs))
        options = pd.concat(options, axis=0)
        options = options.reset_index(drop=True, inplace=False)
        return options

    @staticmethod
    def calculator(options, *args, **kwargs):
        positions = {Variables.Securities.Position.LONG: "supply", Variables.Securities.Position.SHORT: "demand"}
        for position, column in positions.items():
            existing = options.drop(columns=list(positions.values()), inplace=False)
            updated = options["supply"].rename("size")
            dataframe = pd.concat([existing, updated], axis=1)
            dataframe["cashflow"] = dataframe["price"].apply(np.negative) * int(position)
            dataframe["position"] = position
            for greek in ["value", "delta", "gamma", "theta", "rho", "vega"]:
                try: dataframe[greek] = dataframe[greek] * int(position)
                except KeyError: pass
            yield dataframe



