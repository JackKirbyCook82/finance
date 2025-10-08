# -*- coding: utf-8 -*-
"""
Created on Thurs Nov 21 2024
@name:   Prospect Objects
@author: Jack Kirby Cook

"""

import inspect
import numpy as np
import pandas as pd
from itertools import chain
from functools import reduce

from finance.concepts import Concepts, Querys, Securities
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
    if not np.isfinite(number): return "InF"
    elif number > 10 ** 12: return "EsV"
    elif number > 10 ** 9: return f"{number / (10 ** 9):.2f}B%"
    elif number > 10 ** 6: return f"{number / (10 ** 6):.2f}M%"
    elif number > 10 ** 3: return f"{number / (10 ** 3):.2f}K%"
    else: return f"{number:.0f}%"


class ProspectParameters(metaclass=ParameterMeta):
    order = ["strategy"] + list(Querys.Settlement) + list(map(str, Securities.Options)) + ["liquidity", "tau", "spot", "future", "npv"]
    columns = ["tau", "spot", "future", "npv", "liquidity", "priority", "status"]
    index = ["strategy"] + list(map(str, chain(Querys.Settlement, Securities.Options)))
    formatters = {"spot future npv": floating, "liquidity": integer, tuple(map(str, Securities.Options)): floating}
    stacking = Stacking(axis="scenario", columns=["npv", "future", "spot"], layers=list(Concepts.Valuations.Scenario))
    layout = Layout(width=250, space=10, columns=30, rows=30)


class ProspectCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, liquidity, priority, **kwargs):
        super().__init__(*args, **kwargs)
        assert callable(liquidity) and callable(priority)
        self.__liquidity = liquidity
        self.__priority = priority

    def execute(self, valuations, /, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        if self.empty(valuations): return
        settlements = self.keys(valuations, by=Querys.Settlement)
        settlements = ",".join(list(map(str, settlements)))
        prospects = self.calculate(valuations, **kwargs)
        size = self.size(prospects)
        self.console(f"{str(settlements)}[{int(size):.0f}]")
        if self.empty(prospects): return
        yield prospects

    def calculate(self, valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        valuations["liquidity"] = valuations.apply(self.liquidity, axis=1).apply(np.floor).astype(np.int32)
        valuations["status"] = Concepts.Markets.Status.PROSPECT
        valuations["priority"] = valuations.apply(self.priority, axis=1)
        valuations = valuations.where(valuations["liquidity"] >= 1).dropna(how="all", inplace=False)
        valuations = valuations.sort_values("priority", axis=0, ascending=False, inplace=False, ignore_index=False)
        valuations = valuations.reset_index(drop=True, inplace=False)
        return valuations

    @property
    def liquidity(self): return self.__liquidity
    @property
    def priority(self): return self.__priority


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
        assert isinstance(status, (Concepts.Markets.Status, list))
        assert all([isinstance(value, Concepts.Markets.Status) for value in status]) if isinstance(status, list) else True
        super().__init__(*args, **kwargs)
        self.__status = [status] if not isinstance(status, list) else list(status)

    def read(self, *args, **kwargs):
        if not bool(self.table): return
        mask = [self.table["status"] == value for value in self.status]
        mask = reduce(lambda lead, lag: lead | lag, mask)
        dataframe = self.table.take(mask)
        if self.empty(dataframe): return
        return dataframe

    @property
    def status(self): return self.__status


class ProspectWriter(Writer):
    def write(self, dataframe, *args, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        if self.empty(dataframe): return
        columns = list(Querys.Settlement) + list(map(str, Securities.Options))
        self.table.append(dataframe)
        self.table.unique(columns=columns, reverse=True)
        self.table.sort("priority", reverse=True)
        self.table.reindex()







