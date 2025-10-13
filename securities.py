# -*- coding: utf-8 -*-
"""
Created on Tues May 13 2025
@name:   Security Objects
@author: Jack Kirby Cook

"""

import types
import numpy as np
import pandas as pd

from finance.concepts import Querys, Concepts
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


class SecurityCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def execute(self, stocks, options, technicals=None, /, **kwargs):
        assert isinstance(stocks, pd.DataFrame) and isinstance(options, pd.DataFrame)
        assert isinstance(technicals, (pd.DataFrame, types.NoneType))
        if self.empty(options): return
        settlements = self.keys(options, by=Querys.Settlement)
        settlements = ",".join(list(map(str, settlements)))
        securities = self.calculate(stocks, options, **kwargs)
        size = self.size(securities)
        self.console(f"{str(settlements)}[{int(size):.0f}]")
        if all([column in securities.columns for column in ("quoting", "timing")]):
            quoting = set(securities["quoting"].values)
            quoting = list(map(lambda value: str(value).title(), quoting))
            quoting = ",".join(list(quoting))
            timing = securities["timing"].min()
            self.console(f"{str(settlements)}[{quoting}|{timing:%Y-%m-%d %I:%M %p}]")
        if technicals is not None: securities = self.technicals(securities, technicals, **kwargs)
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
        positions = {Concepts.Securities.Position.LONG: "supply", Concepts.Securities.Position.SHORT: "demand"}
        pricing, sizing, greeks = ("ask", "bid"), ("supply", "demand"), ("value", "implied", "delta", "gamma", "theta", "rho", "vega")
        for position, column in positions.items():
            spot = (options["price"].apply(np.negative) * int(position)).rename("spot")
            size = options[column].rename("size")
            dataframe = pd.concat([options, spot, size], axis=1)
            dataframe["position"] = position
            for greek in greeks:
                try: dataframe[greek] = dataframe[greek] * int(position)
                except KeyError: pass
            dataframe = dataframe.drop(columns=list(pricing + sizing))
            yield dataframe

    @staticmethod
    def technicals(securities, technicals, *args, **kwargs):
        assert isinstance(securities, pd.DataFrame) and isinstance(technicals, pd.DataFrame)
        mask = technicals["date"] == technicals["date"].max()
        technicals = technicals.where(mask).dropna(how="all", inplace=False)
        technicals = technicals.drop(columns="date", inplace=False)
        securities = securities.merge(technicals, how="left", on=list(Querys.Symbol), sort=False, suffixes=("", ""))
        return securities





