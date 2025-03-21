# -*- coding: utf-8 -*-
"""
Created on Thurs Nov 21 2024
@name:   Prospect Objects
@author: Jack Kirby Cook

"""

import inspect
import numpy as np
import pandas as pd
from numbers import Number
from functools import reduce
from itertools import count, chain

from finance.variables import Variables, Querys, Securities
from support.mixins import Emptying, Sizing, Partition, Logging
from support.tables import Reader, Writer, Routine, Stacking, Layout
from support.decorators import Decorator
from support.meta import ParameterMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ProspectCalculator", "ProspectRoutine", "ProspectReader", "ProspectWriter", "ProspectParameters"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


def floating(number): return f"{number:.2f}"
def integer(number): return f"{number:.0f}"
def percent(number):
    assert isinstance(number, Number)
    if (100 * number) < 10**3: return f"{number * 100:.2f}%"
    elif (100 * number) < 10**6: return f"{number * 100:.2f}K%"
    elif (100 * number) < 10**9: return f"{number * 100:.2f}M%"
    elif not np.isfinite(percent): return "EsV"
    else: return "InF"


class ProspectParameters(metaclass=ParameterMeta):
    order = list(Querys.Settlement) + list(map(str, Securities.Options)) + ["apy", "npv", "spot", "size"]
    columns = ["apy", "npv", "rev", "exp", "spot", "size", "priority", "status"]
    index = ["order", "valuation", "strategy"] + list(map(str, chain(Querys.Settlement, Securities.Options)))
    formatters = {"apy": percent, "npv rev exp spot": floating, "size": integer, tuple(map(str, Securities.Options)): floating}
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__counter = count(start=1, step=1)

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
        valuations["order"] = [next(self.counter) for _ in range(len(valuations))]
        valuations["status"] = Variables.Markets.Status.PROSPECT
        valuations = valuations.reset_index(drop=True, inplace=False)
        return valuations

    @property
    def counter(self): return self.__counter







