# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 2025
@name:   Stability Objects
@author: Jack Kirby Cook

"""

import pandas as pd
import xarray as xr
from functools import reduce
from collections import OrderedDict as ODict

from finance.variables import Variables, Querys, Securities
from support.calculations import Calculation, Equation, Variable
from support.mixins import Emptying, Sizing, Partition, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StabilityCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class StabilityEquation(Equation):
    pass


class StabilityCalculation(Calculation, equation=StabilityEquation):
    def execute(self, *args, **kwargs): pass


class StabilityCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__calculation = StabilityCalculation(*args, **kwargs)

    def execute(self, valuations, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame) and isinstance(valuations, pd.DataFrame)
        if self.empty(valuations): return
        for settlement, primary in self.partition(valuations, options, by=Querys.Settlement):
            secondary = self.alignment(options, by=settlement)
            results = self.calculate(primary, secondary, *args, **kwargs)
            size = self.size(results)
            self.console(f"{str(settlement)}[{int(size):.0f}]")
            if self.empty(results): continue
            yield results

    def calculate(self, valuations, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame) and isinstance(valuations, pd.DataFrame)
        exposures = self.exposures(options, *args, **kwargs)
        orders = self.orders(valuations, *args, **kwargs)

        print(exposures)
        print(orders)
        raise Exception()

    @staticmethod
    def exposures(options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        index = list(Querys.Settlement) + list(Variables.Securities.Security) + ["strike", "order"]
        exposures = options.assign(order=0).set_index(index, drop=True, inplace=False)
        exposures = exposures["exposure"].squeeze()
        exposures = xr.DataArray.from_series(exposures).fillna(0)
        function = lambda content, axis: content.squeeze(axis)
        exposures = reduce(function, list(Querys.Settlement), exposures)
        return exposures

    @staticmethod
    def orders(valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        parameters = dict(id_vars=list(Querys.Settlement) + ["order"], value_name="strike", var_name="security")
        header = list(Querys.Settlement) + list(map(str, Securities.Options)) + ["order"]
        index = list(Querys.Settlement) + list(Variables.Securities.Security) + ["strike", "order"]
        function = lambda security: pd.Series(ODict(Securities.Options[security].items()))
        valuations["order"] = list(range(1, len(valuations) + 1))
        orders = valuations[header].droplevel(level=1, axis=1)
        orders = pd.melt(orders, **parameters)
        orders = pd.concat([orders, orders["security"].apply(function)], axis=1)
        orders = orders.dropna(subset="strike", how="all", inplace=False)
        orders = orders.assign(exposure=1).set_index(index, drop=True, inplace=False)
        orders = orders["exposure"].squeeze()
        orders = xr.DataArray.from_series(orders).fillna(0)
        function = lambda content, axis: content.squeeze(axis)
        orders = reduce(function, list(Querys.Settlement), orders)
        return orders













