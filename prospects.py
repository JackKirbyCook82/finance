# -*- coding: utf-8 -*-
"""
Created on Thurs Nov 21 2024
@name:   Prospect Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from abc import ABC
from functools import reduce
from itertools import product, count
from collections import OrderedDict as ODict

from finance.variables import Variables, Querys
from support.mixins import Emptying, Sizing, Logging, Sourcing
from support.tables import Reader, Routine, Writer, Table
from support.meta import ParametersMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ProspectCalculator", "ProspectReader", "ProspectDiscarding", "ProspectAltering", "ProspectWriter", "ProspectTable", "ProspectHeader", "ProspectLayout"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class ProspectTable(Table, ABC, datatype=pd.DataFrame): pass
class ProspectParameters(metaclass=ParametersMeta):
    scenarios = {Variables.Valuations.ARBITRAGE: [Variables.Scenarios.MINIMUM, Variables.Scenarios.MAXIMUM]}
    context = ["valuation", "strategy"]
    prospect = ["order", "priority", "status"]
    variants = {Variables.Valuations.ARBITRAGE: ["apy", "npv", "cost"]}
    invariants = ["underlying", "size", "current"]
    options = list(map(str, Variables.Securities.Options))
    stocks = list(map(str, Variables.Securities.Stocks))
    contract = list(map(str, Querys.Contract))


class ProspectHeader(ProspectParameters):
    def __iter__(self): return iter(self.index + self.columns)
    def __init__(self, *args, valuation, context, contract, options, scenarios, variants, invariants, prospect, **kwargs):
        assert valuation in Variables.Valuations
        index, columns = list(context + contract + options), list(invariants + prospect)
        scenarios, variants = list(scenarios[valuation]), list(variants[valuation])
        self.name = kwargs.get("name", self.__class__.__name__)
        self.columns = list(product(variants, scenarios)) + list(product(columns, [""]))
        self.index = list(product(index, [""]))
        self.scenarios = scenarios
        self.variants = variants
        self.valuation = valuation


class ProspectLayout(ProspectParameters):
    def __init__(self, *args, valuation, context, contract, options, scenarios, variants, **kwargs):
        order = list(contract + context + options + variants[valuation] + ["size", "status"])
        formats = dict(status=lambda status: str(status), size=lambda size: f"{size:.0f}")
        formats["apy"] = lambda value: (f"{value * 100:.02f}%" if value < 10 else "EsV") if np.isfinite(value) else "InF"
        variants, scenarios = list(variants[valuation]), list(scenarios[valuation])
        generator = lambda key: product([key], scenarios) if key in variants else product([key], [""])
        self.name = kwargs.get("name", self.__class__.__name__)
        self.formats = {column: function for key, function in formats.items() for column in generator(key)}
        self.order = [column for key in order for column in generator(key)]
        self.numbers = lambda value: f"{value:.02f}"
        self.width = kwargs.get("width", 250)
        self.columns = kwargs.get("columns", 30)
        self.rows = kwargs.get("rows", 30)


class ProspectCalculator(Logging, Sizing, Emptying, Sourcing):
    def __init__(self, *args, priority, header, **kwargs):
        assert callable(priority)
        super().__init__(*args, **kwargs)
        self.counter = count(start=1, step=1)
        self.priority = priority
        self.header = header

    def execute(self, valuations, *args, **kwargs):
        if self.empty(valuations): return
        for contract, dataframe in self.source(valuations, *args, query=Querys.Contract, **kwargs):
            prospects = self.calculate(dataframe, *args, **kwargs)
            prospects = prospects.reindex(columns=list(self.header), fill_value=np.NaN)
            size = self.size(prospects)
            string = f"Calculated: {repr(self)}|{str(contract)}[{size:.0f}]"
            self.logger.info(string)
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
        return prospects


class ProspectReader(Reader):
    def __init__(self, *args, status, **kwargs):
        assert isinstance(status, list) and all([isinstance(value, Variables.Status) for value in status])
        super().__init__(*args, **kwargs)
        self.status = status

    def take(self, status):
        mask = self.table["status"] == status
        dataframe = self.table.take(mask)
        size = self.size(dataframe)
        string = f"{str(status.name)}: {repr(self)}[{size:.0f}]"
        self.logger.info(string)
        return dataframe

    def read(self, *args, **kwargs):
        if not bool(self.table): return
        dataframes = list(map(self.take, self.status))
        return pd.concat(dataframes, axis=0)


class ProspectDiscarding(Routine):
    def __init__(self, *args, status, **kwargs):
        assert isinstance(status, list) and all([isinstance(value, Variables.Status) for value in status])
        super().__init__(*args, **kwargs)
        self.status = status

    def take(self, status):
        mask = self.table["status"] == status
        dataframe = self.table.take(mask)
        size = self.size(dataframe)
        string = f"{str(status.name)}: {repr(self)}[{size:.0f}]"
        self.logger.info(string)

    def routine(self, *args, **kwargs):
        if not bool(self.table): return
        for status in self.status:
            self.take(status)


class ProspectWriter(Writer):
    def __init__(self, *args, status, **kwargs):
        assert isinstance(status, Variables.Status)
        super().__init__(*args, **kwargs)
        self.status = status

    def detach(self, query):
        if not bool(self.table): return
        mask = [self.table[key] == value for key, value in query.items()]
        mask = reduce(lambda lead, lag: lead & lag, mask)
        mask = mask & (self.table["status"] == self.status)
        dataframe = self.table.take(mask)
        size = self.size(dataframe)
        string = f"Detached: {repr(self)}[{size:.0f}]"
        self.logger.info(string)

    def attach(self, content):
        self.table.append(content)
        size = self.size(content)
        string = f"Appended: {repr(self)}[{size:.0f}]"
        self.logger.info(string)

    def write(self, prospects, *args, **kwargs):
        for query, content in self.source(prospects, *args, query=self.query, **kwargs):
            if self.empty(content): continue
            content["status"] = self.status
            self.detach(query)
            self.attach(content)
        self.table.sort("priority", reverse=True)
        self.table.reindex()


class ProspectAltering(Routine):
    def __init__(self, *args, protocols=[], **kwargs):
        assert isinstance(protocols, (list, tuple))
        protocols = [protocols] if isinstance(protocols, tuple) else list(protocols)
        assert all([isinstance(protocol, tuple) and len(protocol) == 3 for protocol in protocols])
        super().__init__(*args, **kwargs)
        self.protocols = protocols

    def routine(self, *args, **kwargs):
        for name, status, function in self.protocols:
            mask = function(self.table)
            self.table.modify(mask, "status", status)



