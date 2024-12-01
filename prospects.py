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

from finance.variables import Variables, Querys
from support.mixins import Emptying, Sizing, Logging, Sourcing
from support.tables import Reader, Routine, Writer, Table, View
from support.meta import ParametersMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ProspectCalculator", "ProspectReader", "ProspectDiscarding", "ProspectAltering", "ProspectWriter", "ProspectTable"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class ProspectParameters(metaclass=ParametersMeta):
    scenarios = {Variables.Valuations.ARBITRAGE: [Variables.Scenarios.MINIMUM, Variables.Scenarios.MAXIMUM]}
    context = ["valuation", "strategy"]
    prospect = ["order", "priority", "status"]
    variate = {Variables.Valuations.ARBITRAGE: ["apy", "npv", "cost"]}
    invariant = ["underlying", "size", "current"]
    options = list(map(str, Variables.Securities.Options))
    stocks = list(map(str, Variables.Securities.Stocks))
    contract = list(map(str, Variables.Contract))


class ProspectLayout(ProspectParameters):
    def __init__(self, parameters, *args, valuation, **kwargs):
        assert valuation in Variables.Valuations
        order = parameters["contract"] + parameters["context"] + parameters["options"]
        order = order + parameters["variant"][valuation] + ["size", "status"]
        self.name = kwargs.get("name", self.__class__.__name__)
        self.formats = dict(status=lambda status: str(status), size=lambda size: f"{size:.0f}")
        self.formats["apy"] = lambda column: (f"{column * 100:.02f}%" if column < 10 else "EsV") if np.isfinite(column) else "InF"
        self.numbers = lambda column: f"{column:.02f}"
        self.valuation = valuation
        self.order = order


class ProspectHeader(ProspectParameters):
    def __iter__(self): return iter(self.index + self.columns)
    def __init__(self, parameters, *args, valuation, **kwargs):
        assert valuation in Variables.Valuations
        index = parameters["context"] + parameters["contract"] + parameters["options"]
        scenarios = parameters["scenario"][valuation]
        variate = parameters["variant"][valuation]
        invariant = parameters["invariant"]
        self.name = kwargs.get("name", self.__class__.__name__)
        self.columns = list(product(variate, scenarios)) + list(product(invariant, [""]))
        self.index = list(product(index, [""]))
        self.scenarios = scenarios
        self.variate = variate
        self.valuation = valuation


class ProspectView(View, ABC, datatype=pd.DataFrame, layouttype=ProspectLayout): pass
class ProspectTable(Table, ABC, datatype=pd.DataFrame, viewtype=ProspectView, headertype=ProspectHeader): pass


class ProspectCalculator(Logging, Sizing, Emptying, Sourcing):
    def __init__(self, *args, priority, **kwargs):
        assert callable(priority)
        super().__init__(*args, **kwargs)
        self.counter = count(start=1, step=1)
        self.priority = priority

    def execute(self, valuations, *args, **kwargs):
        if self.empty(valuations): return
        for contract, dataframe in self.source(valuations, *args, query=Querys.Contract, **kwargs):
            prospects = self.calculate(dataframe, *args, **kwargs)
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
        prospects["status"] = None
        return prospects


class ProspectReader(Reader):
    def __init__(self, *args, status, **kwargs):
        assert isinstance(status, list) and all([isinstance(value, Variables.Status) for value in status])
        super().__init__(*args, **kwargs)
        self.status = status

    def remove(self, status):
        mask = self.table[:, "status"] == status
        dataframe = self.table.extract(mask)
        size = self.size(dataframe)
        string = f"{str(self.status)}: {repr(self)}[{size:.0f}]"
        self.logger.info(string)
        return dataframe

    def read(self, *args, **kwargs):
        dataframes = list(map(self.remove, self.status))
        return pd.concat(dataframes, axis=0)


class ProspectDiscarding(Routine):
    def __init__(self, *args, status, **kwargs):
        assert isinstance(status, list) and all([isinstance(value, Variables.Status) for value in status])
        super().__init__(*args, **kwargs)
        self.status = status

    def remove(self, status):
        mask = self.table[:, "status"] == status
        dataframe = self.table.extract(mask)
        size = self.size(dataframe)
        string = f"{str(self.status)}: {repr(self)}[{size:.0f}]"
        self.logger.info(string)

    def routine(self, *args, **kwargs):
        for status in self.status:
            self.remove(status)


class ProspectWriter(Writer):
    def __init__(self, *args, status, **kwargs):
        assert isinstance(status, Variables.Status)
        super().__init__(*args, **kwargs)
        self.status = status

    def remove(self, query):
        mask = [self[:, key] == value for key, value in query.items()]
        mask = reduce(lambda lead, lag: lead & lag, mask)
        mask = mask & self[:, "status"] == self.status
        dataframe = self.table.extract(mask)
        size = self.size(dataframe)
        string = f"Removed: {repr(self)}[{size:.0f}]"
        self.logger.info(string)

    def append(self, content):
        self.table.combine(content)
        size = self.size(content)
        string = f"Appended: {repr(self)}[{size:.0f}]"
        self.logger.info(string)

    def write(self, prospects, *args, **kwargs):
        for query, content in self.source(prospects, *args, query=self.query, **kwargs):
            content["status"] = self.status
            self.remove(query)
            self.append(content)
        self.table.sort("priority", reverse=True)
        self.table.reset()


class ProspectAltering(Routine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def routine(self, *args, **kwargs):
        pass



