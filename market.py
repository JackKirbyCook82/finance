# -*- coding: utf-8 -*-
"""
Created on Tues Mar 18 2025
@name:   Market Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from functools import reduce
from abc import ABC, abstractmethod

from finance.variables import Querys, Variables, Securities
from support.mixins import Emptying, Sizing, Partition, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["AcquisitionCalculator", "DivestitureCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class MarketCalculator(Sizing, Emptying, Partition, Logging, ABC, title="Calculated"):
    def __init__(self, *args, liquidity, priority, **kwargs):
        super().__init__(*args, **kwargs)
        assert callable(liquidity) and callable(priority)
        self.__liquidity = liquidity
        self.__priority = priority

    def execute(self, valuations, securities, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame) and isinstance(securities, pd.DataFrame)
        if self.empty(valuations): return
        for settlement, primary, secondary in self.alignment(valuations, securities, by=Querys.Settlement):
            dataframe = self.calculate(primary, secondary, *args, **kwargs)
            size = self.size(dataframe)
            self.console(f"{str(settlement)}[{int(size):.0f}]")
            if self.empty(dataframe): continue
            yield dataframe

    def alignment(self, valuations, securities, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame) and isinstance(securities, pd.DataFrame)
        for partition, primary in self.partition(valuations, *args, **kwargs):
            mask = [securities[key] == value for key, value in iter(partition)]
            mask = reduce(lambda lead, lag: lead & lag, list(mask))
            secondary = securities.where(mask)
            yield partition, primary, secondary

    def calculate(self, valuations, securities, *args, **kwargs):
        valuations["priority"] = valuations.apply(self.priority, axis=1)
        securities["size"] = securities.apply(self.liquidity, axis=1).apply(np.round).astype(np.int32)
        valuations = valuations.sort_values("priority", axis=0, ascending=False, inplace=False, ignore_index=False)

        print("\n", valuations, "\n")
        print(securities, "\n")

        interest = self.interest(valuations, *args, **kwargs)
        market = self.market(securities, *args, **kwargs)
        market = market[market.index.isin(interest)]
        header = list(Querys.Settlement) + list(map(str, Securities.Options))
        valuations["size"] = valuations[header].apply(self.equilibrium, axis=1, market=market)
        mask = valuations[("size", "")] > 0
        valuations = valuations.where(mask).dropna(how="all", inplace=False)

        print(valuations, "\n")
        raise Exception()

    @staticmethod
    def interest(valuations, *args, **kwargs):
        parameters = dict(id_vars=list(Querys.Settlement), value_name="strike", var_name="security")
        header = list(Querys.Settlement) + list(map(str, Securities.Options))
        interest = valuations[header].droplevel(level=1, axis=1)
        interest = pd.melt(interest, **parameters)
        mask = interest["strike"].isna()
        interest = interest.where(~mask).dropna(how="all", inplace=False)
        interest = pd.MultiIndex.from_frame(interest)
        return interest

    @staticmethod
    def market(securities, *args, **kwargs):
        security = lambda cols: str(Securities([cols["instrument"], cols["option"], cols["position"]]))
        header = list(Querys.Settlement) + list(Variables.Securities.Security) + ["strike", "size", "owned", "shares"]
        market = securities[header]
        market["security"] = market.apply(security, axis=1)
        index = list(Querys.Settlement) + ["security", "strike"]
        market = market[index + ["size", "owned", "shares"]].set_index(index, drop=True, inplace=False)
        return market

    @staticmethod
    @abstractmethod
    def equilibrium(valuation, *args, market, **kwargs): pass

    @property
    def liquidity(self): return self.__liquidity
    @property
    def priority(self): return self.__priority


class AcquisitionCalculator(MarketCalculator):
    @staticmethod
    def equilibrium(valuation, *args, market, **kwargs):
        parameters = dict(id_vars=list(Querys.Settlement), value_name="strike", var_name="security")
        valuation = valuation.droplevel(level=1)
        valuation = valuation.to_frame().transpose()
        valuation = pd.melt(valuation, **parameters)
        mask = valuation["strike"].isna()
        valuation = valuation.where(~mask).dropna(how="all", inplace=False)
        index = pd.MultiIndex.from_frame(valuation)
        quantity = market.loc[index]
        quantity["size"] = quantity["size"].min()
        quantity = quantity.reindex(market.index).fillna(0)
        market["size"] = market["size"] - quantity["size"]
        return quantity.loc[index, "size"].min()


class DivestitureCalculator(MarketCalculator):
    @staticmethod
    def equilibrium(valuation, *args, market, **kwargs):
        print("\n", valuation, "\n")
        print(market, "\n")
        raise Exception()




