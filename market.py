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

from finance.variables import Variables, Querys, Securities
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

    def execute(self, valuations, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame) and isinstance(valuations, pd.DataFrame)
        if self.empty(valuations): return
        for settlement, primary, secondary in self.alignment(valuations, options, by=Querys.Settlement):
            results = self.calculate(primary, secondary, *args, **kwargs)
            size = self.size(results)
            self.console(f"{str(settlement)}[{int(size):.0f}]")
            if self.empty(results): continue
            yield results

    def alignment(self, valuations, securities, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame) and isinstance(securities, pd.DataFrame)
        for partition, primary in self.partition(valuations, *args, **kwargs):
            mask = [securities[key] == value for key, value in iter(partition)]
            mask = reduce(lambda lead, lag: lead & lag, list(mask))
            secondary = securities.where(mask)
            yield partition, primary, secondary

    def calculate(self, valuations, options, *args, **kwargs):
        valuations["priority"] = valuations.apply(self.priority, axis=1)
        options["size"] = options.apply(self.liquidity, axis=1)
        valuations = valuations.sort_values("priority", axis=0, ascending=False, inplace=False, ignore_index=False)
        demand = self.demand(valuations, *args, **kwargs)
        supply = self.supply(options, *args, **kwargs)
        supply = supply[supply.index.isin(demand)]

        print("\n", valuations, "\n")
        print(options, "\n")
        print(supply, "\n")
        raise Exception()

        header = list(Querys.Settlement) + list(map(str, Securities.Options))
        valuations["size"] = valuations[header].apply(self.equilibrium, axis=1, available=supply)
        mask = valuations[("size", "")] > 0
        valuations = valuations.where(mask).dropna(how="all", inplace=False)
        valuations = valuations.reset_index(drop=True, inplace=False)
        return valuations

    @staticmethod
    def demand(valuations, *args, **kwargs):
        parameters = dict(id_vars=list(Querys.Settlement), value_name="strike", var_name="security")
        header = list(Querys.Settlement) + list(map(str, Securities.Options))
        demand = valuations[header].droplevel(level=1, axis=1)
        demand = pd.melt(demand, **parameters)
        mask = demand["strike"].isna()
        demand = demand.where(~mask).dropna(how="all", inplace=False)
        demand = pd.MultiIndex.from_frame(demand)
        return demand

    @staticmethod
    def supply(securities, *args, **kwargs):
        security = lambda cols: str(Securities([cols["instrument"], cols["option"], cols["position"]]))
        header = list(Querys.Settlement) + list(Variables.Securities.Security) + ["strike"]
        index = list(Querys.Settlement) + ["security", "strike"]
        columns = ["size", "quantity"] if "quantity" in securities.columns else ["size"]
        supply = securities[header + columns]
        supply["security"] = supply.apply(security, axis=1)
        supply = supply[index + columns].set_index(index, drop=True, inplace=False)
        return supply

    @staticmethod
    def equilibrium(valuation, *args, available, **kwargs):
        pass

    # @staticmethod
    # def equilibrium(valuation, *args, available, **kwargs):
    #     parameters = dict(id_vars=list(Querys.Settlement), value_name="strike", var_name="security")
    #     valuation = valuation.droplevel(level=1)
    #     valuation = valuation.to_frame().transpose()
    #     valuation = pd.melt(valuation, **parameters)
    #     mask = valuation["strike"].isna()
    #     valuation = valuation.where(~mask).dropna(how="all", inplace=False)
    #     index = pd.MultiIndex.from_frame(valuation)
    #     quantity = available.loc[index]
    #     quantity["size"] = quantity["size"].min()
    #     quantity = quantity.reindex(available.index).fillna(0).astype(np.int32)
    #     available["size"] = available["size"] - quantity["size"]
    #     return quantity.loc[index, "size"].min().astype(np.int32)

    @property
    def liquidity(self): return self.__liquidity
    @property
    def priority(self): return self.__priority



