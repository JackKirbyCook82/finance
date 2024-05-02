# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 2024
@name:   Technical Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from abc import ABC

from support.calculations import Variable, Domain, Equation, Calculation, Calculator
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


class StochasticDomain(Domain, domain={"xi": "price", "xli": "lowest", "xhi": "highest"}): pass
class StochasticEquation(Equation):
    xki = Variable("oscillator", np.float32, lambda xi, xli, xhi: (xi - xli) * 100 / (xhi - xli))


class TechnicalCalculation(Calculation, ABC, fields=["technical"]): pass
class StatisticCalculation(Calculation, technical=Technicals.STATISTIC):
    @staticmethod
    def execute(bars, *args, period, **kwargs):
        yield bars["price"].pct_change(1).rolling(period).mean().rename("trend")
        yield bars["price"].pct_change(1).rolling(period).std().rename("volatility")

class StochasticCalculation(Calculation, technical=Technicals.STOCHASTIC, equation=StochasticEquation, domain=StochasticDomain):
    def execute(self, bars, *args, period, **kwargs):
        low = bars["lowest"].rolling(period).min()
        high = bars["highest"].rolling(period).max()
        bars = pd.concat([bars, low, high], axis=1)
        domain = self.domain(bars)
        equation = self.equation(domain)
        yield equation.xki


class TechnicalCalculator(Calculator, Processor, calculation=TechnicalCalculation):
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
        for fields, calculation in self.calculations.items():
            variable = str(fields["technical"].name).lower()
            dataframe = calculation(bars, *args, **kwargs)
            if not self.size(dataframe):
                continue
            yield variable, dataframe



