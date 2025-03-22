# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 2025
@name:   Stability Objects
@author: Jack Kirby Cook

"""

import pandas as pd

from finance.variables import Querys
from support.mixins import Emptying, Sizing, Partition, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StabilityCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class StabilityCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def execute(self, valuations, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame) and isinstance(valuations, pd.DataFrame)
        if self.empty(valuations): return
        for settlement, primary, secondary in self.partition(valuations, options, by=Querys.Settlement):
            secondary = self.alignment(options, by=settlement)
            results = self.calculate(primary, secondary, *args, **kwargs)
            size = self.size(results)
            self.console(f"{str(settlement)}[{int(size):.0f}]")
            if self.empty(results): continue
            yield results

    def calculate(self, valuations, options, *args, **kwargs):
        pass


