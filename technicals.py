# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 2024
@name:   Technical Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from abc import ABC
from collections import OrderedDict as ODict

from support.calculations import Variable, Equation, Calculation, Calculator
from support.query import Header, Query
from support.files import Files

from finance.variables import Technicals

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["BarsFile", "StatisticFile", "StochasticFile", "TechnicalCalculator"]
__copyright__ = "Copyright 2024, Jack Kirby Cook"
__license__ = "MIT License"


technical_index = {"date": np.datetime64}
bars_columns = {"high": np.float32, "low": np.float32, "open": np.float32, "close": np.float32, "price": np.float32, "volume": np.float32}
bars_header = Header(pd.DataFrame, index=list(technical_index.keys()), columns=list(bars_columns.keys()), ascending="date")
statistic_columns = {"price": np.float32, "trend": np.float32, "volatility": np.float32}
statistic_header = Header(pd.DataFrame, index=list(technical_index.keys()), columns=list(statistic_columns.keys()))
stochastic_columns = {"price": np.float32, "oscillator": np.float32}
stochastic_header = Header(pd.DataFrame, index=list(technical_index.keys()), columns=list(stochastic_columns.keys()))
technical_headers = ODict(list({"statistic": statistic_header, "stochastic": stochastic_header}.items()))


class BarsFile(Files.Dataframe, variable="bars", index=technical_index, columns=bars_columns): pass
class StatisticFile(Files.Dataframe, variable="statistic", index=technical_index, columns=statistic_columns): pass
class StochasticFile(Files.Dataframe, variable="stochastic", index=technical_index, columns=stochastic_columns): pass


class TechnicalEquation(Equation): pass
class StochasticEquation(TechnicalEquation):
    xki = Variable("xki", "oscillator", np.float32, function=lambda xi, xli, xhi: (xi - xli) * 100 / (xhi - xli))
    xi = Variable("xi", "price", np.float32, position=0, locator="price")
    xli = Variable("xli", "lowest", np.float32, position=0, locator="lowest")
    xhi = Variable("xhi", "highest", np.float32, position=0, locator="highest")


class TechnicalCalculation(Calculation, ABC, fields=["technical"]): pass
class StatisticCalculation(TechnicalCalculation, technical=Technicals.STATISTIC):
    @staticmethod
    def execute(bars, *args, period, **kwargs):
        yield bars["price"]
        yield bars["price"].pct_change(1).rolling(period).mean().rename("trend")
        yield bars["price"].pct_change(1).rolling(period).std().rename("volatility")

class StochasticCalculation(TechnicalCalculation, technical=Technicals.STOCHASTIC, equation=StochasticEquation):
    def execute(self, bars, *args, period, **kwargs):
        equation = self.equation(*args, **kwargs)
        lowest = bars["low"].rolling(period).min().rename("lowest")
        highest = bars["high"].rolling(period).max().rename("highest")
        bars = pd.concat([bars, lowest, highest], axis=1)
        yield bars["price"]
        yield equation.xki(bars)


class TechnicalCalculator(Calculator, calculations=TechnicalCalculation):
    @Query()
    def execute(self, bars, *args, **kwargs):
        technicals = ODict(list(self.calculate(bars, *args, **kwargs)))
        yield technicals

    def calculate(self, bars, *args, **kwargs):
        for variables, calculation in self.calculations.items():
            variable = str(variables["technical"].name).lower()
            dataframe = calculation(bars, *args, **kwargs)
            yield variable, dataframe



