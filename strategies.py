# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Strategy Objects
@author: Jack Kirby Cook

"""

import numpy as np

from support.calculations import Calculation
from support.processes import Calculator
from support.pipelines import Processor
from support.files import Files

from finance.variables import Securities, Strategies

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StrategyFile", "StrategyCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


strategy_index = {security: str for security in list(map(str, Securities))} | {"strategy": str, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
strategy_columns = {}


class StrategyFile(Files.Dataframe, variable="strategies", index=strategy_index, columns=strategy_columns):
    pass


class StrategyCalculation(Calculation):
    pass


class StrategyCalculator(Calculator, Processor):
    def execute(self, contents, *args, **kwargs):
        pass



