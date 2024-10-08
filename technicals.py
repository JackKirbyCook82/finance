# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 2024
@name:   Technical Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd
from abc import ABC

from finance.variables import Variables, Symbol
from support.calculations import Variable, Equation, Calculation
from support.mixins import Emptying, Sizing, Logging, Pipelining
from support.meta import RegistryMeta, ParametersMeta
from support.files import File

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TechnicalCalculator", "BarsFile", "StatisticFile", "StochasticFile"]
__copyright__ = "Copyright 2024, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class TechnicalParameters(metaclass=ParametersMeta):
    types = {"ticker": str, "volume": np.int64} | {column: np.float32 for column in ("price", "open", "close", "high", "low")}
    types.update({"trend": np.float32, "volatility": np.float32, "oscillator": np.float32})
    dates = {"date": "%Y%m%d"}
    queryname = lambda filename: Symbol.fromstring(filename)
    filename = lambda queryname: Symbol.tostring(queryname)


class TechnicalVariables(object):
    axes = {Variables.Querys.HISTORY: ["date", "ticker"], Variables.Querys.SYMBOL: ["ticker"]}
    data = {Variables.Technicals.STATISTIC: ["price", "trend", "volatility"], Variables.Technicals.STOCHASTIC: ["price", "oscillator"]}

    def __init__(self, *args, technical, **kwargs):
        assert technical in self.data.keys()
        self.history = self.axes[Variables.Querys.HISTORY]
        self.symbol = self.axes[Variables.Querys.SYMBOL]
        self.index = self.history
        self.columns = self.data[technical]
        self.header = self.index + self.columns


class TechnicalFile(File, datatype=pd.DataFrame, **dict(TechnicalParameters)): pass
class BarsFile(TechnicalFile, variable=Variables.Technicals.BARS): pass
class StatisticFile(TechnicalFile, variable=Variables.Technicals.STATISTIC): pass
class StochasticFile(TechnicalFile, variable=Variables.Technicals.STOCHASTIC): pass


class TechnicalEquation(Equation): pass
class StochasticEquation(TechnicalEquation):
    xk = Variable("xk", "oscillator", np.float32, function=lambda x, xl, xh: (x - xl) * 100 / (xh - xl))
    xh = Variable("xh", "highest", np.float32, position=0, locator="highest")
    xl = Variable("xl", "lowest", np.float32, position=0, locator="lowest")
    x = Variable("x", "price", np.float32, position=0, locator="price")


class TechnicalCalculation(Calculation, ABC, metaclass=RegistryMeta): pass
class StatisticCalculation(TechnicalCalculation, register=Variables.Technicals.STATISTIC):
    @staticmethod
    def execute(bars, *args, period, **kwargs):
        assert (bars["ticker"].to_numpy()[0] == bars["ticker"]).all()
        yield from iter([bars["ticker"], bars["date"], bars["price"]])
        yield bars["price"].pct_change(1).rolling(period).mean().rename("trend")
        yield bars["price"].pct_change(1).rolling(period).std().rename("volatility")

class StochasticCalculation(TechnicalCalculation, equation=StochasticEquation, register=Variables.Technicals.STOCHASTIC):
    def execute(self, bars, *args, period, **kwargs):
        assert (bars["ticker"].to_numpy()[0] == bars["ticker"]).all()
        equation = self.equation(*args, **kwargs)
        lowest = bars["low"].rolling(period).min().rename("lowest")
        highest = bars["high"].rolling(period).max().rename("highest")
        bars = pd.concat([bars, lowest, highest], axis=1)
        yield from iter([bars["ticker"], bars["date"], bars["price"]])
        yield equation.xk(bars)


class TechnicalCalculator(Pipelining, Sizing, Emptying, Logging):
    def __init__(self, *args, technical, **kwargs):
        Pipelining.__init__(self, *args, **kwargs)
        Logging.__init__(self, *args, **kwargs)
        self.__variables = TechnicalVariables(*args, technical=technical, **kwargs)
        self.__calculation = TechnicalCalculation[technical](*args, **kwargs)
        self.__technical = technical

    def execute(self, bars, *args, **kwargs):
        assert isinstance(bars, pd.DataFrame)
        for symbol, dataframe in self.symbols(bars):
            parameters = dict(ticker=symbol.ticker)
            technicals = self.calculate(dataframe, *args, **parameters, **kwargs)
            size = self.size(technicals)
            string = f"Calculated: {repr(self)}|{str(symbol)}[{size:.0f}]"
            self.logger.info(string)
            if self.empty(technicals): continue
            yield technicals

    def calculate(self, bars, *args, **kwargs):
        assert isinstance(bars, pd.DataFrame)
        bars = bars.sort_values("date", ascending=False, inplace=False)
        technicals = self.calculation(bars, *args, **kwargs)
        assert isinstance(technicals, pd.DataFrame)
        technicals = technicals if bool(technicals) else pd.DataFrame(columns=self.variables.header)
        return technicals

    def symbols(self, bars):
        assert isinstance(bars, pd.DataFrame)
        for symbol, dataframe in bars.groupby(self.variables.symbol):
            if self.empty(dataframe): continue
            yield Symbol(*symbol), dataframe

    @property
    def calculation(self): return self.__calculations
    @property
    def variables(self): return self.__variables
    @property
    def technical(self): return self.__technical



