# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 2024
@name:   Technical Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from collections import OrderedDict as ODict

from support.calculations import Equation, Calculation, Calculator
from support.pipelines import Processor
from support.files import Files

from finance.variables import Technicals

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["BarFile", "TechnicalFile", "TechnicalCalculator"]
__copyright__ = "Copyright 2024, Jack Kirby Cook"
__license__ = "MIT License"


history_index = {"date": np.datetime64}
bars_columns = {"high": np.float32, "low": np.float32, "open": np.float32, "close": np.float32, "price": np.float32, "volume": np.float32}
technicals_columns = {"trend": np.float32, "volatility": np.float32, "oscillator": np.float32}


class BarFile(Files.Dataframe, variable="bars", index=history_index, columns=bars_columns): pass
class TechnicalFile(Files.Dataframe, variable="technicals", index=history_index, columns=technicals_columns): pass


class StochasticEquation(Equation):
    xki = lambda xi, xli, xhi: (xi - xli) * 100 / (xhi - xli)


class TechnicalCalculation(Calculation, fields=["technical"]): pass
class StatisticCalculation(Calculation, technical=Technicals.STATISTIC):
    @staticmethod
    def execute(bars, *args, period, **kwargs):
        pass

#        yield bars["price"].pct_change(1).rolling(period).mean()
#        yield bars["price"].pct_change(1).rolling(period).std()

class StochasticCalculation(Calculation, technical=Technicals.STOCHASTIC):
    @staticmethod
    def execute(bars, *args, period, **kwargs):
        pass

#        equation = StochasticEquation(domain=["xi", "xli", "xhi"])
#        low = bars["low"].rolling(period).min()
#        high = bars["high"].rolling(period).max()
#        yield equation.xki(bars["price"], low, high)


class TechnicalCalculator(Calculator, Processor, calculations=ODict(list(TechnicalCalculation)), title="Calculated"):
    def execute(self, contents, *args, **kwargs):
        bars = contents["bars"]
        assert isinstance(bars, pd.DataFrame)
        if self.empty(bars):
            return
        technicals = {variable: technical for variable, technical in self.calculate(bars, *args, **kwargs)}
        technicals = pd.concat(list(technicals.values()), axis=1)
        yield contents | technicals

    def calculate(self, bars, *args, **kwargs):
        assert isinstance(bars, pd.DataFrame)
        for technical, calculation in self.calculations.items():
            variable = str(technical.name).lower()
            dataframe = calculation(bars, *args, **kwargs)
            if not self.size(dataframe):
                continue
            yield variable, dataframe



