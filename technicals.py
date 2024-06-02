# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 2024
@name:   Technical Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from abc import ABC
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

from finance.variables import Technicals
from support.calculations import Variable, Equation, Calculation
from support.files import FileDirectory, FileQuery, FileData
from support.pipelines import Processor

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["BarsFile", "StatisticFile", "StochasticFile", "TechnicalCalculator", "TechnicalHeader"]
__copyright__ = "Copyright 2024, Jack Kirby Cook"
__license__ = "MIT License"


Header = ntuple("Header", "index columns")
technical_index = {"date": np.datetime64}
bars_columns = {"high": np.float32, "low": np.float32, "open": np.float32, "close": np.float32, "price": np.float32, "volume": np.float32}
statistic_columns = {"price": np.float32, "trend": np.float32, "volatility": np.float32}
stochastic_columns = {"price": np.float32, "oscillator": np.float32}
technical_headers = {"bars": Header(technical_index, bars_columns), "statistic": Header(technical_index, statistic_columns), "stochastic": Header(technical_index, stochastic_columns)}
bars_data = FileData.Dataframe(header=technical_index | bars_columns)
statistic_data = FileData.Dataframe(header=technical_index | statistic_columns)
stochastic_data = FileData.Dataframe(header=technical_index | stochastic_columns)
ticker_query = FileQuery("ticker", str.upper, str.upper)


class BarsFile(FileDirectory, variable="bars", query=ticker_query, data=bars_data): pass
class StatisticFile(FileDirectory, variable="statistic", query=ticker_query, data=statistic_data): pass
class StochasticFile(FileDirectory, variable="stochastic", query=ticker_query, data=stochastic_data): pass


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


class TechnicalCalculator(Processor):
    def __init__(self, *args, technicals=[], name=None, **kwargs):
        assert isinstance(technicals, list) and all([strategy in list(Technicals) for strategy in technicals])
        super().__init__(*args, name=name, **kwargs)
        calculations = {variables["technical"]: calculation for variables, calculation in ODict(list(TechnicalCalculation)).items() if variables["technical"] in technicals}
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


class TechnicalHeader(Processor):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.__technicals = dict(technical_headers)

    def execute(self, contents, *args, **kwargs):
        technicals = {technical: contents[technical] for technical in self.technicals if technical in contents.keys()}
        technicals = ODict(list(self.calculate(technicals, *args, **kwargs)))
        yield contents | dict(technicals)

    def calculate(self, technicals, *args, **kwargs):
        for technical, dataframe in technicals.items():
            dataframe = dataframe.sort_values("date", axis=0, ascending=True, inplace=False)
            header = self.technicals[technical]

            print(dataframe)
            print(header.index)

            dataframe = dataframe.set_index(header.index, drop=True, inplace=False)
            dataframe = dataframe[header.columns]
            yield technical, dataframe

    @property
    def technicals(self): return self.__technicals



