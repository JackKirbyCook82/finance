# -*- coding: utf-8 -*-
"""
Created on Thurs Nov 21 2024
@name:   Prospect Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from functools import reduce
from itertools import product, count

from finance.variables import Variables, Querys
from support.mixins import Emptying, Sizing, Logging, Separating
from support.tables import Reader, Routine, Writer, Table

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ProspectCalculator", "ProspectReader", "ProspectDiscarding", "ProspectProtocols", "ProspectWriter", "ProspectTable", "ProspectHeader", "ProspectLayout"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class ProspectTable(Table): pass
class ProspectParameters(object):
    scenarios = {Variables.Valuations.ARBITRAGE: [Variables.Scenarios.MINIMUM, Variables.Scenarios.MAXIMUM]}
    variants = {Variables.Valuations.ARBITRAGE: ["apy", "npv", "cost"]}
    invariants = ["underlying", "size", "current"]
    prospect = ["order", "priority", "status"]
    context = ["valuation", "strategy"]
#    options = list(map(str, Variables.Securities.Options))
#    stocks = list(map(str, Variables.Securities.Stocks))
    contract = list(map(str, Querys.Contract))


class ProspectHeader(ProspectParameters):
    def __iter__(self): return iter(self.index + self.columns)
    def __new__(cls, *args, valuation, **kwargs):
        assert valuation in Variables.Valuations
        scenarios = list(cls.scenarios[valuation])
        variants = list(cls.variants[valuation])
        index = list(cls.context + cls.contract + cls.options)
        columns = list(cls.invariants + cls.prospect)
        instance = super().__new__(cls)
        instance.name = cls.__name__
        instance.index = list(product(index, [""]))
        instance.columns = list(product(variants, scenarios)) + list(product(columns, [""]))
        instance.transform = ("scenario", variants)
        instance.scenarios = scenarios
        instance.variants = variants
        instance.valuation = valuation
        return instance


class ProspectLayout(ProspectParameters):
    def __new__(cls, *args, valuation, **kwargs):
        assert valuation in Variables.Valuations
        scenarios = list(cls.scenarios[valuation])
        variants = list(cls.variants[valuation])
        order = list(cls.contract + cls.context + cls.options + variants + ["size", "status"])
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


class ProspectCalculator(Separating, Sizing, Emptying, Logging):
    def __init__(self, *args, priority, header, **kwargs):
        assert callable(priority)
        super().__init__(*args, **kwargs)
        self.__counter = count(start=1, step=1)
        self.__query = Querys.Contract
        self.__priority = priority
        self.__header = header

    def execute(self, valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        if self.empty(valuations): return
        for parameters, dataframe in self.separate(valuations, *args, fields=self.fields, **kwargs):
            contract = self.query(parameters)
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

    @property
    def fields(self): return list(self.__query)
    @property
    def priority(self): return self.__priority
    @property
    def counter(self): return self.__counter
    @property
    def header(self): return self.__header
    @property
    def query(self): return self.__query


class ProspectReader(Reader, query=Querys.Contract):
    def __init__(self, *args, status, **kwargs):
        assert isinstance(status, list) and all([isinstance(value, Variables.Status) for value in status])
        super().__init__(*args, **kwargs)
        self.__status = status

    def take(self, status):
        mask = self.table["status"] == status
        dataframe = self.table.take(mask)
        size = self.size(dataframe)
        title = str(status.name).lower().title()
        string = f"{str(title)}: {repr(self)}[{size:.0f}]"
        self.logger.info(string)
        return dataframe

    def read(self, *args, **kwargs):
        if not bool(self.table): return
        dataframes = list(map(self.take, self.status))
        return pd.concat(dataframes, axis=0)

    @property
    def status(self): return self.__status


class ProspectDiscarding(Routine, query=Querys.Contract):
    def __init__(self, *args, status, **kwargs):
        assert isinstance(status, list) and all([isinstance(value, Variables.Status) for value in status])
        super().__init__(*args, **kwargs)
        self.__status = status

    def take(self, status):
        mask = self.table["status"] == status
        dataframe = self.table.take(mask)
        size = self.size(dataframe)
        title = str(status.name).lower().title()
        string = f"{str(title)}: {repr(self)}[{size:.0f}]"
        self.logger.info(string)

    def invoke(self, *args, **kwargs):
        if not bool(self.table): return
        for status in self.status:
            self.take(status)

    @property
    def status(self): return self.__status


class ProspectProtocols(Routine, query=Querys.Contract):
    def __init__(self, *args, protocols={}, **kwargs):
        assert isinstance(protocols, dict)
        assert all([callable(protocol) for protocol in protocols.keys()])
        assert all([status in Variables.Status for status in protocols.values()])
        super().__init__(*args, **kwargs)
        self.__protocols = dict(protocols)

    def invoke(self, *args, **kwargs):
        if not bool(self.table): return
        for protocol, status in self.protocols.items():
            mask = protocol(self.table)
            self.table.modify(mask, "status", status)

    @property
    def protocols(self): return self.__protocols


class ProspectWriter(Writer, query=Querys.Contract):
    def __init__(self, *args, status, **kwargs):
        assert isinstance(status, Variables.Status)
        super().__init__(*args, **kwargs)
        self.__status = status

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
        for parameters, content in self.separate(prospects, *args, fields=self.fields, **kwargs):
            if self.empty(content): continue
            query = self.query(parameters)
            content["status"] = self.status
            self.detach(query)
            self.attach(content)
        self.table.sort("priority", reverse=True)
        self.table.reindex()

    @property
    def status(self): return self.__status



