# -*- coding: utf-8 -*-
"""
Created on Thurs Nov 21 2024
@name:   Prospect Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from abc import ABC
from itertools import product, count

from finance.variables import Variables, Querys
from support.mixins import Emptying, Sizing, Logging, Sourcing
from support.tables import Reader, Writer, Table, View
from support.meta import ParametersMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ProspectCalculator", "ProspectWriter", "ProspectReader", "ProspectTable"]
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


class ProspectFormat(ProspectParameters):
    def __init__(self, parameters, *args, valuation, **kwargs):
        assert valuation in Variables.Valuations
        order = parameters["contract"] + parameters["context"] + parameters["options"]
        order = order + parameters["variant"][valuation] + ["size", "status"]
        self.name = kwargs.get("name", self.__class__.__name__)
        self.format = dict(status=lambda status: str(status), size=lambda size: f"{size:.0f}")
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


class ProspectView(View, ABC, datatype=pd.DataFrame, formattype=ProspectFormat): pass
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
        prospects["status"] = Variables.Status.PROSPECT
        return prospects


class ProspectWriter(Writer, query=Querys.Contract):
    def write(self, prospects, *args, **kwargs):
        prospects = self.prospects(prospects, *args, **kwargs)
        self.table.combine(prospects)
        self.table.sort("priority", reverse=True)
        self.table.reset()

    def prospects(self, prospects, *args, **kwargs):
        assert isinstance(prospects, pd.DataFrame)
        assert (prospects["status"] == Variables.Status.PROSPECT).all()
        mask = self.table[:, "prospect"] == Variables.Status.PROSPECT
        existing = self.table.dataframe[mask]
        index = ["valuation", "strategy"] + list(map(str, Querys.Contract)) + list(map(str, Variables.Securities.Options))
        columns = list(prospects.columns)
        overlap = existing.merge(prospects, on=index, how="inner", suffixes=("_", ""))[columns]
        prospects = pd.concat([prospects, overlap], axis=0)
        prospects = prospects.drop_duplicates(index, keep="last", inplace=False)
        return prospects


class ProspectReader(Reader, query=Querys.Contract):
    def read(self, *args, **kwargs):
        pass












