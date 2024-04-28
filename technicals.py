# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 2024
@name:   Technical Objects
@author: Jack Kirby Cook

"""

import numpy as np

from support.calculations import Calculation
from support.processes import Calculator
from support.pipelines import Processor
from support.files import Files

from finance.variables import Technicals


__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["BarFile", "StatisticFile", "TechnicalCalculator"]
__copyright__ = "Copyright 2024, Jack Kirby Cook"
__license__ = "MIT License"


history_index = {"ticker": str, "date": np.datetime64}
bars_columns = {"high": np.float32, "low": np.float32, "open": np.float32, "close": np.float32, "price": np.float32, "price": np.float32, "volume": np.float32}
stats_columns = {"trend": np.float32, "volatility": np.float32}


class BarFile(Files.Dataframe, variable="bars", index=history_index, columns=bars_columns): pass
class StatisticFile(Files.Dataframe, variable="statistics", index=history_index, columns=stats_columns): pass


class TechnicalCalculation(Calculation):
    pass


class TechnicalCalculator(Calculator, Processor):
    def execute(self, contents, *args, **kwargs):
        pass



