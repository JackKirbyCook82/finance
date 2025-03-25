# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 2025
@name:   Stability Objects
@author: Jack Kirby Cook

"""

import pandas as pd

from finance.variables import Querys
from support.calculations import Calculation, Equation
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

    def execute(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if self.empty(options): return
        for settlement, dataframe in self.partition(options, by=Querys.Settlement):
            results = self.calculate(dataframe, *args, **kwargs)
            size = self.size(results)
            self.console(f"{str(settlement)}[{int(size):.0f}]")
            if self.empty(results): continue
            yield results

    def calculate(self, options, *args, **kwargs):

        print(options)
        raise Exception()


class StabilityFilter(Sizing, Emptying, Partition, Logging, title="Filtered"):
    def execute(self, valuations, options, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame) and isinstance(options, pd.DataFrame)
        if self.empty(valuations): return
        for settlement, primary in self.partition(valuations, by=Querys.Settlement):
            secondary = self.alignment(options, by=settlement)
            results = self.calculate(primary, secondary, *args, **kwargs)
            size = self.size(results)
            self.console(f"{str(settlement)}[{int(size):.0f}]")
            if self.empty(results): continue
            yield results

    def calculate(self, valuations, options, *args, **kwargs):
        pass







