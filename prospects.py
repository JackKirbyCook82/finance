# -*- coding: utf-8 -*-
"""
Created on Thurs Nov 21 2024
@name:   Prospect Objects
@author: Jack Kirby Cook

"""

import inspect
import numpy as np
import pandas as pd
from functools import reduce
from itertools import count, chain

from finance.variables import Variables, Querys, Securities
from support.mixins import Emptying, Sizing, Partition, Logging
from support.tables import Reader, Writer, Routine, Stacking, Layout
from support.decorators import Decorator
from support.meta import MappingMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ProspectCalculator", "ProspectRoutine", "ProspectReader", "ProspectWriter", "ProspectParameters"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class ProspectParameters(metaclass=MappingMeta):
    order = list(Querys.Settlement) + list(map(str, Securities.Options)) + ["apy", "npv", "rev", "exp", "spot", "share", "size", "status"]
    columns = ["apy", "npv", "rev", "exp", "spot", "share", "size", "current", "priority", "status"]
    index = ["order", "valuation", "strategy"] + list(map(str, chain(Querys.Settlement, Securities.Options)))
    percent = lambda value: (f"{value * 100:.2f}%" if value < 10 else "EsV") if np.isfinite(value) else "InF"
    financial, floating, integer = lambda value: f"{value:.2f}", lambda value: f"{value:.2f}", lambda value: f"{value:.0f}"
    formatting = {"apy": percent, "npv rev exp spot share": financial, "size": integer, "status": str, tuple(map(str, Securities.Options)): floating}
    stacking = Stacking(axis="scenario", columns=["apy", "npv", "rev", "exp"], layers=list(Variables.Valuations.Scenario))
    layout = Layout(width=250, space=10, columns=30, rows=30)


class ProspectRoutine(Routine):
    def __init__(self, *args, protocol, **kwargs):
        super().__init__(*args, **kwargs)
        self.__protocol = protocol

    def routine(self, *args, **kwargs):
        if not self.table: return
        predicate = lambda value: isinstance(value, Decorator) and "status" in value
        for name, protocol in inspect.getmembers(self.protocol, predicate=predicate):
            mask = protocol(self.table)
            self.table.modify(mask, "status", protocol["status"])

    @property
    def protocol(self): return self.__protocol


class ProspectReader(Reader):
    def __init__(self, *args, status, **kwargs):
        assert isinstance(status, (Variables.Markets.Status, list))
        assert all([isinstance(value, Variables.Markets.Status) for value in status]) if isinstance(status, list) else True
        super().__init__(*args, **kwargs)
        self.__status = [status] if not isinstance(status, list) else list(status)

    def read(self, *args, **kwargs):
        if not bool(self.table): return
        mask = [self.table["status"] == value for value in self.status]
        mask = reduce(lambda lead, lag: lead | lag, mask)
        dataframes = self.table.take(mask)
        if self.empty(dataframes): return
        for settlement, dataframes in self.partition(dataframes, by=Querys.Settlement):
            for status, dataframe in dataframes.groupby("status", sort=False):
                if self.empty(dataframe): continue
                size = self.size(dataframe)
                status = str(status).lower()
                self.console(f"{str(status)}[{int(size):.0f}]", title="Detached")
        return dataframe

    @property
    def status(self): return self.__status


class ProspectWriter(Writer):
    def write(self, dataframes, *args, **kwargs):
        if self.empty(dataframes): return
        for settlement, dataframe in self.partition(dataframes, by=Querys.Settlement):
            if self.empty(dataframe): continue
            self.table.append(dataframe)
            size = self.size(dataframe)
            status = str(Variables.Markets.Status.PROSPECT).lower()
            self.console(f"{str(settlement)}|{str(status)}[{int(size):.0f}]", title="Attached")
        columns = list(Querys.Settlement) + list(map(str, Securities.Options))
        self.table.unique(columns=columns, reverse=True)
        self.table.sort("priority", reverse=True)
        self.table.reindex()


class ProspectCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, priority, liquidity, header, **kwargs):
        assert callable(priority) and callable(liquidity)
        super().__init__(*args, **kwargs)
        self.__counter = count(start=1, step=1)
        self.__liquidity = liquidity
        self.__priority = priority
        self.__header = header

    def execute(self, valuations, securities, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame) and isinstance(securities, pd.DataFrame)
        if self.empty(valuations): return
        for settlement, primary, secondary in self.alignment(valuations, securities, by=Querys.Settlement):
            prospects = self.calculate(primary, secondary, *args, **kwargs)
            size = self.size(prospects)
            self.console(f"{str(settlement)}[{int(size):.0f}]")
            if self.empty(prospects): continue
            yield prospects

    def alignment(self, valuations, securities, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame) and isinstance(securities, pd.DataFrame)
        for partition, primary in self.partition(valuations, *args, **kwargs):
            mask = [securities[key] == value for key, value in iter(partition)]
            mask = reduce(lambda lead, lag: lead & lag, list(mask))
            secondary = securities.where(mask)
            yield partition, primary, secondary

    def calculate(self, valuations, securities, *args, **kwargs):
        valuations["priority"] = valuations.apply(self.priority, axis=1)
        valuations["size"] = valuations.apply(self.liquidity, axis=1)
        securities["size"] = securities.apply(self.liquidity, axis=1)
        valuations = valuations.sort_values("priority", axis=0, ascending=False, inplace=False, ignore_index=False)
        demand = self.demand(valuations, *args, **kwargs)
        supply = self.supply(securities, *args, **kwargs)
        supply = supply[supply.index.isin(demand)]
        header = list(Querys.Settlement) + list(map(str, Securities.Options))
        valuations["size"] = valuations[header].apply(self.quantify, axis=1, securities=supply)
        mask = valuations[("size", "")] > 0
        valuations = valuations.where(mask).dropna(how="all", inplace=False)
        valuations["order"] = [next(self.counter) for _ in range(len(valuations))]
        valuations["status"] = Variables.Markets.Status.PROSPECT
        valuations = valuations.reindex(columns=list(self.header), fill_value=np.NaN)
        valuations = valuations.reset_index(drop=True, inplace=False)
        return valuations

    @staticmethod
    def quantify(valuation, *args, securities, **kwargs):
        parameters = dict(id_vars=list(Querys.Settlement), value_name="strike", var_name="security")
        valuation = valuation.droplevel(level=1)
        valuation = valuation.to_frame().transpose()
        valuation = pd.melt(valuation, **parameters)
        mask = valuation["strike"].isna()
        valuation = valuation.where(~mask).dropna(how="all", inplace=False)
        index = pd.MultiIndex.from_frame(valuation)
        quantity = securities.loc[index]
        quantity["size"] = quantity["size"].min()
        quantity = quantity.reindex(securities.index).fillna(0).astype(np.int32)
        securities["size"] = securities["size"] - quantity["size"]
        return quantity.loc[index, "size"].min().astype(np.int32)

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
        header = list(Querys.Settlement) + list(Variables.Securities.Security) + ["strike", "size"]
        supply = securities[header]
        supply["security"] = supply.apply(security, axis=1)
        index = list(Querys.Settlement) + ["security", "strike"]
        supply = supply[index + ["size"]].set_index(index, drop=True, inplace=False)
        return supply

    @property
    def liquidity(self): return self.__liquidity
    @property
    def priority(self): return self.__priority
    @property
    def counter(self): return self.__counter
    @property
    def header(self): return self.__header







