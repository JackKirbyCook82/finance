# -*- coding: utf-8 -*-
"""
Created on Weds Mar 5 2025
@name:   Market Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from functools import reduce

from finance.variables import Variables, Querys, Securities
from support.mixins import Emptying, Sizing, Partition, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["MarketCalculator", "MarketFilter"]
__copyright__ = "Copyright 2025, Jack Kirby Cook"
__license__ = "MIT License"


class MarketCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, liquidity, **kwargs):
        assert callable(liquidity)
        super().__init__(*args, **kwargs)
        self.__header = list(Querys.Settlement) + ["security", "strike"]
        self.__liquidity = liquidity

    def partition(self, valuations, securities, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame) and isinstance(securities, pd.DataFrame)
        for partition, primary in super().partition(valuations, *args, **kwargs):
            mask = [securities[key] == value for key, value in iter(partition)]
            mask = reduce(lambda lead, lag: lead & lag, list(mask))
            secondary = securities.where(mask)
            yield partition, (primary, secondary)

    def execute(self, valuations, securities, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame) and isinstance(securities, pd.DataFrame)
        if self.empty(valuations): return
        for settlement, (primary, secondary) in self.partition(valuations, securities, by=Querys.Settlement):
            market = self.calculate(primary, secondary, *args, **kwargs)
            size = self.size(market)
            self.console(f"{str(settlement)}[{int(size):.0f}]")
            if self.empty(market): continue
            yield market

    def calculate(self, valuations, securities, *args, **kwargs):
        demand = self.demand(valuations, *args, **kwargs)
        supply = self.supply(securities, *args, **kwargs)
        supply = supply[supply.index.isin(demand.index)]
        market = pd.concat([supply, demand], axis=1)
        return market

    def supply(self, securities, *args, **kwargs):
        security = lambda cols: str(Securities([cols["instrument"], cols["option"], cols["position"]]))
        header = list(Querys.Settlement) + list(Variables.Securities.Security) + ["strike", "size"]
        supply = securities[header]
        supply["supply"] = supply.apply(self.liquidity, axis=1).astype(np.int32)
        supply["security"] = supply.apply(security, axis=1)
        supply = supply[self.header + ["supply"]].set_index(self.header, drop=True, inplace=False)
        return supply

    def demand(self, valuations, *args, **kwargs):
        parameters = dict(id_vars=list(Querys.Settlement) + ["size"], value_name="strike", var_name="security")
        function = lambda column: column in valuations.columns
        header = list(Querys.Settlement) + list(filter(function, map(str, Securities.Options))) + ["size"]
        demand = valuations[header].droplevel(level=1, axis=1)
        demand = pd.melt(demand, **parameters)
        mask = demand["strike"].isna()
        demand = demand.where(~mask).dropna(how="all", inplace=False)
        demand["demand"] = demand.apply(self.liquidity, axis=1).astype(np.int32)
        demand = demand.drop("size", axis=1, inplace=False)
        demand = demand[self.header + ["demand"]].groupby(self.header).agg(np.sum)
        return demand

    @property
    def liquidity(self): return self.__liquidity
    @property
    def header(self): return self.__header


class MarketFilter(Sizing, Emptying, Partition, Logging, title="Filtered"):
    def execute(self, prospects, market, *args, **kwargs):
        assert isinstance(prospects, pd.DataFrame) and isinstance(market, pd.DataFrame)
        if self.empty(prospects): return
        for settlement, dataframe in self.partition(prospects, by=Querys.Settlement):
            prior = self.size(dataframe)
            dataframe = self.calculate(dataframe, market, *args, **kwargs)
            post = self.size(dataframe)
            string = f"{str(settlement)}[{prior:.0f}|{post:.0f}]"
            self.console(string)
            if self.empty(dataframe): continue
            yield dataframe

    def calculate(self, prospects, market, *args, **kwargs):
        print(prospects, "\n")

        columns = list(Querys.Settlement) + list(map(str, Securities.Options)) + ["size"]
        market = market.assign(quantity=0)
        prospects["quantity"] = prospects[columns].apply(self.quantify, axis=1, market=market)

        print(prospects, "\n")
        raise Exception()

    def quantify(self, prospect, market, *args, **kwargs):
        parameters = dict(id_vars=list(Querys.Settlement), value_name="strike", var_name="security")
        prospect = prospect.droplevel(level=1)
        size = prospect.pop("size")
        prospect = prospect.to_frame().transpose()
        prospect = pd.melt(prospect, **parameters)
        mask = prospect["strike"].isna()
        prospect = prospect.where(~mask).dropna(how="all", inplace=False)
        index = pd.MultiIndex.from_frame(prospect)

        print(index, "\n")
        print(market, "\n")
        raise Exception()

#        prospect = market.loc[index, :]
#        supply = prospect["supply"] - prospect["quantity"]
#        demand = prospect["demand"] - prospect["quantity"]
#        equilibrium = pd.DataFrame([supply, demand]).min().rename("equilibrium")
#        quantity = equilibrium.min()
#        assert (equilibrium == quantity).all()
#        assert (equilibrium <= size).all()
#        equilibrium = equilibrium.reindex(market.index).fillna(0).astype(np.int32)
#        market["quantity"] = market["quantity"] + equilibrium
#        return quantity




