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

from finance.variables import Symbol, Technicals
from support.calculations import Variable, Equation, Calculation
from support.pipelines import Processor
from support.parsers import Header
from support.files import File

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TechnicalFiles", "TechnicalHeaders", "TechnicalCalculator"]
__copyright__ = "Copyright 2024, Jack Kirby Cook"
__license__ = "MIT License"


technical_index = {"date": np.datetime64}
bars_columns = {"ticker": str, "high": np.float32, "low": np.float32, "open": np.float32, "close": np.float32, "price": np.float32, "volume": np.float32}
statistic_columns = {"ticker": str, "price": np.float32, "trend": np.float32, "volatility": np.float32}
stochastic_columns = {"ticker": str, "price": np.float32, "oscillator": np.float32}


class BarsFile(File, variable="bars", query=Symbol, datatype=pd.DataFrame, header=technical_index | bars_columns): pass
class StatisticFile(File, variable="statistic", query=Symbol, datatype=pd.DataFrame, header=technical_index | statistic_columns): pass
class StochasticFile(File, variable="stochastic", query=Symbol, datatype=pd.DataFrame, header=technical_index | stochastic_columns): pass
class BarsHeader(Header, variable="bars", datatype=pd.DataFrame, axes={"index": technical_index, "columns": bars_columns}): pass
class StatisticHeader(Header, variable="statistic", datatype=pd.DataFrame, axes={"index": technical_index, "columns": statistic_columns}): pass
class StochasticHeader(Header, variable="stochastic", datatype=pd.DataFrame, axes={"index": technical_index, "columns": stochastic_columns}): pass


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
        yield bars["ticker"]
        yield bars["price"]
        yield bars["price"].pct_change(1).rolling(period).mean().rename("trend")
        yield bars["price"].pct_change(1).rolling(period).std().rename("volatility")

class StochasticCalculation(TechnicalCalculation, technical=Technicals.STOCHASTIC, equation=StochasticEquation):
    def execute(self, bars, *args, period, **kwargs):
        equation = self.equation(*args, **kwargs)
        lowest = bars["low"].rolling(period).min().rename("lowest")
        highest = bars["high"].rolling(period).max().rename("highest")
        bars = pd.concat([bars, lowest, highest], axis=1)
        yield bars["ticker"]
        yield bars["price"]
        yield equation.xki(bars)


class TechnicalCalculator(Processor):
    def __init__(self, *args, calculations=[], name=None, **kwargs):
        assert isinstance(calculations, list) and all([strategy in list(Technicals) for strategy in calculations])
        super().__init__(*args, name=name, **kwargs)
        calculations = {variables["technical"]: calculation for variables, calculation in ODict(list(TechnicalCalculation)).items() if variables["technical"] in calculations}
        self.__calculations = {str(technical.name).lower(): calculation(*args, **kwargs) for technical, calculation in calculations.items()}

    def execute(self, contents, *args, **kwargs):
        bars = contents["bars"]
        assert isinstance(bars, pd.DataFrame)
        technicals = ODict(list(self.calculate(bars, *args, **kwargs)))
        yield contents | dict(technicals)

    def calculate(self, bars, *args, **kwargs):
        for technical, calculation in self.calculations.items():
            dataframe = calculation(bars, *args, **kwargs)
            yield technical, dataframe

    @property
    def calculations(self): return self.__calculations


class TechnicalFiles(object):
    Bars = BarsFile
    Statistic = StatisticFile
    Stochastic = StochasticFile

class TechnicalHeaders(object):
    Bars = BarsHeader
    Statistic = StatisticHeader
    Stochastic = StochasticHeader



