# -*- coding: utf-8 -*-
"""
Created on Tues May 13 2025
@name:   Security Objects
@author: Jack Kirby Cook

"""

import pandas as pd
from collections import namedtuple as ntuple

from finance.variables import Querys, Variables
from support.mixins import Emptying, Sizing, Partition, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StockCalculator", "OptionCalculator", "SecurityCalculator"]
__copyright__ = "Copyright 2025, Jack Kirby Cook"
__license__ = "MIT License"


class StockCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, pricing, **kwargs):
        assert callable(pricing)
        super().__init__(*args, **kwargs)
        self.__pricing = pricing

    def execute(self, stocks, technicals, *args, **kwargs):
        assert isinstance(stocks, pd.DataFrame)
        if self.empty(stocks): return
        symbols = self.keys(stocks, by=Querys.Symbol)
        symbols = ",".join(list(map(str, symbols)))
        stocks = self.calculate(stocks, technicals, *args, **kwargs)
        size = self.size(stocks)
        self.console(f"{str(symbols)}[{int(size):.0f}]")
        if self.empty(stocks): return
        yield stocks

    def calculate(self, stocks, technicals, *args, **kwargs):
        assert isinstance(stocks, pd.DataFrame) and isinstance(technicals, pd.DataFrame)
        mask = technicals["date"] == technicals["date"].max()
        technicals = technicals.where(mask).dropna(how="all", inplace=False)
        technicals = technicals.drop(columns="date", inplace=False)
        price = stocks.apply(self.pricing, axis=1).rename("price")
        stocks = pd.concat([stocks, price], axis=1)
        stocks["instrument"] = Variables.Securities.Instrument.STOCK
        stocks = stocks.merge(technicals, how="left", on=list(Querys.Symbol), sort=False, suffixes=("", ""))
        stocks = stocks.reset_index(drop=True, inplace=False)
        return stocks

    @property
    def pricing(self): return self.__pricing


class OptionCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, pricing, **kwargs):
        assert callable(pricing)
        super().__init__(*args, **kwargs)
        self.__pricing = pricing

    def execute(self, options, stocks, *args, **kwargs):
        assert isinstance(options, pd.DataFrame) and isinstance(stocks, pd.DataFrame)
        if self.empty(options): return
        settlements = self.keys(options, by=Querys.Settlement)
        settlements = ",".join(list(map(str, settlements)))
        options = self.calculate(options, stocks, *args, **kwargs)
        size = self.size(options)
        self.console(f"{str(settlements)}[{int(size):.0f}]")
        if self.empty(options): return
        yield options

    def calculate(self, options, stocks, *args, **kwargs):
        assert isinstance(options, pd.DataFrame) and isinstance(stocks, pd.DataFrame)
        underlying = stocks.apply(lambda series: series["price"], axis=1).rename("underlying")
        stocks = pd.concat([stocks, underlying], axis=1)
        price = options.apply(self.pricing, axis=1).rename("price")
        options = pd.concat([options, price], axis=1)
        options["instrument"] = Variables.Securities.Instrument.OPTION
        header = set(Querys.Symbol) | set([column for column in stocks.columns if column not in options.columns])
        options = options.merge(stocks[list(header)], how="left", on=list(Querys.Symbol), sort=False, suffixes=("", ""))
        options = options.reset_index(drop=True, inplace=False)
        return options

    @property
    def pricing(self): return self.__pricing


class SecurityCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, pricing, **kwargs):
        assert callable(pricing)
        super().__init__(*args, **kwargs)
        Header = ntuple("Header", "size market limit")
        self.__headers = {Variables.Securities.Position.LONG: Header("supply", "ask", "bid"), Variables.Securities.Position.SHORT: Header("demand", "bid", "ask")}
        self.__pricing = pricing

    def execute(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if self.empty(options): return
        settlements = self.keys(options, by=Querys.Settlement)
        settlements = ",".join(list(map(str, settlements)))
        securities = self.calculate(options, *args, **kwargs)
        size = self.size(securities)
        self.console(f"{str(settlements)}[{int(size):.0f}]")
        if self.empty(securities): return
        yield securities

    def calculate(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        options = list(self.calculator(options, *args, **kwargs))
        options = pd.concat(options, axis=0)
        options = options.reset_index(drop=True, inplace=False)
        return options

    def calculator(self, options, *args, **kwargs):
        for position, header in self.headers.items():
            generator = zip(header._fields, header)
            dataframes = [options[value].rename(key) for key, value in generator]
            dataframe = pd.concat([options] + dataframes, axis=1)
            function = lambda column: lambda series: series[column] * int(position)
            greeks = {column: function(column) for column in ("value", "delta", "gamma", "theta", "rho", "vega")}
            pricing = {"cashflow": lambda series: - self.pricing(series) * int(position)}
            dataframe = dataframe.assign(**pricing, **greeks, position=position)
            yield dataframe

    @property
    def pricing(self): return self.__pricing
    @property
    def headers(self): return self.__headers


