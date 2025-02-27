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
from itertools import product, count

from finance.variables import Variables, Querys, Securities
from support.mixins import Emptying, Sizing, Partition, Logging
from support.tables import Reader, Writer, Routine, Table
from support.decorators import Decorator

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ProspectCalculator", "ProspectRoutine", "ProspectReader", "ProspectWriter", "ProspectTable", "ProspectHeader", "ProspectLayout"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class ProspectTable(Table): pass
class ProspectParameters(object):
    scenarios = {Variables.Valuations.Valuation.ARBITRAGE: [Variables.Valuations.Scenario.MINIMUM, Variables.Valuations.Scenario.MAXIMUM]}
    variants = {Variables.Valuations.Valuation.ARBITRAGE: ["apy", "npv", "cost"]}
    invariants = ["underlying", "size", "current"]
    prospect = ["order", "priority", "status"]
    context = ["valuation", "strategy"]
    options = list(map(str, Securities.Options))
    stocks = list(map(str, Securities.Stocks))
    contract = list(map(str, Querys.Settlement))


class ProspectHeader(ProspectParameters):
    def __iter__(self): return iter(self.index + self.columns)
    def __new__(cls, *args, valuation, **kwargs):
        assert valuation in Variables.Valuations.Valuation
        scenarios = list(cls.scenarios[valuation])
        variants = list(cls.variants[valuation])
        index = list(cls.context + cls.contract + cls.options)
        columns = list(cls.invariants + cls.prospect)
        instance = super().__new__(cls)
        instance.name = cls.__name__
        instance.index = list(product(index, [""]))
        instance.columns = list(product(variants, scenarios)) + list(product(columns, [""]))
        instance.pivot = ("scenario", variants)
        instance.unpivot = ("scenario", variants)
        instance.scenarios = scenarios
        instance.variants = variants
        instance.valuation = valuation
        return instance


class ProspectLayout(ProspectParameters):
    def __new__(cls, *args, valuation, **kwargs):
        assert valuation in Variables.Valuations.Valuation
        scenarios = list(cls.scenarios[valuation])
        variants = list(cls.variants[valuation])
        order = list(cls.contract + cls.context + cls.options + variants + ["underlying", "size", "status"])
        formats = dict(status=lambda status: str(status), size=lambda size: f"{size:.0f}")
        formats["apy"] = lambda value: (f"{value * 100:.02f}%" if value < 10 else "EsV") if np.isfinite(value) else "InF"
        generator = lambda key: product([key], scenarios) if key in variants else product([key], [""])
        instance = super().__new__(cls)
        instance.name = cls.__name__
        instance.formats = {column: function for key, function in formats.items() for column in generator(key)}
        instance.order = [column for key in order for column in generator(key)]
        instance.numbers = lambda value: f"{value:.02f}"
        instance.width = kwargs.get("width", 250)
        instance.columns = kwargs.get("columns", 30)
        instance.rows = kwargs.get("rows", 30)
        instance.valuation = valuation
        return instance


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
        self.obsolete(dataframes, *args, **kwargs)
        self.prospect(dataframes, *args, **kwargs)
        columns = list(Querys.Settlement) + list(map(str, Securities.Options))
        self.table.unique(columns=columns, reverse=True)
        self.table.sort("priority", reverse=True)
        self.table.reindex()

    def prospect(self, dataframes, *args, **kwargs):
        if self.empty(dataframes): return
        for settlement, dataframe in self.partition(dataframes, by=Querys.Settlement):
            if self.empty(dataframe): continue
            self.table.append(dataframe)
            size = self.size(dataframe)
            status = str(Variables.Markets.Status.PROSPECT).lower()
            self.console(f"{str(settlement)}|{str(status)}[{int(size):.0f}]", title="Attached")

    def obsolete(self, dataframes, *args, **kwargs):
        if not bool(self.table): return
        if self.empty(dataframes): return
        for settlement in self.groups(dataframes, by=Querys.Settlement):
            mask = [self.table[key] == value for key, value in settlement.items()]
            mask = reduce(lambda lead, lag: lead & lag, mask)
            mask = mask & (self.table["status"] != Variables.Markets.Status.OBSOLETE)
            self.table.modify(mask, "status", Variables.Markets.Status.OBSOLETE)
            dataframe = self.table.image(mask)
            size = self.size(dataframe)
            status = str(Variables.Markets.Status.OBSOLETE).lower()
            self.console(f"{str(settlement)}|{str(status)}[{int(size):.0f}]", title="Modified")


class ProspectCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, priority, header, **kwargs):
        assert callable(priority)
        super().__init__(*args, **kwargs)
        self.__counter = count(start=1, step=1)
        self.__priority = priority
        self.__header = header

    def execute(self, valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        if self.empty(valuations): return
        for settlement, dataframe in self.partition(valuations, by=Querys.Settlement):
            prospects = self.calculate(dataframe, *args, **kwargs)
            size = self.size(prospects)
            self.console(f"{str(settlement)}[{int(size):.0f}]")
            if self.empty(prospects): continue
            yield prospects

    def calculate(self, valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        prospects = valuations.assign(order=[next(self.counter) for _ in range(len(valuations))])
        prospects["order"] = prospects["order"].astype(np.int64)
        prospects["priority"] = prospects.apply(self.priority, axis=1)
        parameters = dict(ascending=False, inplace=False, ignore_index=False)
        prospects = prospects.sort_values("priority", axis=0, **parameters)
        prospects["status"] = np.NaN
        prospects = prospects.reindex(columns=list(self.header), fill_value=np.NaN)
        prospects["status"] = Variables.Markets.Status.PROSPECT
        return prospects

    @property
    def priority(self): return self.__priority
    @property
    def counter(self): return self.__counter
    @property
    def header(self): return self.__header




