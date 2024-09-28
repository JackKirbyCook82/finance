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
from support.meta import RegistryMeta, ParametersMeta
from support.files import File

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TechnicalCalculator"]
__copyright__ = "Copyright 2024, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class TechnicalParameters(metaclass=ParametersMeta):
    filename = lambda symbol: str(symbol.ticker).upper()
    bars = {"ticker": str, "volume": np.int64} | {column: np.float32 for column in ("price", "open", "close", "high", "low")}
    stochastic = {"trend": np.float32, "volatility": np.float32}
    statistic = {"oscillator": np.float32}
    dates = {"date": "%Y%m%d"}


class TechnicalVariables(object):
    data = {Variables.Technicals.STATISTIC: ["price", "trend", "volatility"], Variables.Technicals.STOCHASTIC: ["price", "oscillator"]}
    axes = {Variables.Querys.HISTORY: ["date", "ticker"], Variables.Querys.SYMBOL: ["ticker"]}

    def __init__(self, *args, technical, **kwargs):
        self.index = self.axes[Variables.Querys.HISTORY]
        self.columns = self.data[technical]
        self.symbol = self.axes[Variables.Querys.SYMBOL]
        self.header = self.index + self.columns


class TechnicalFile(File, datatype=pd.DataFrame, dates=TechnicalParameters.dates, filename=TechnicalParameters.filename): pass
class BarsFile(TechnicalFile, variable=Variables.Technicals.BARS, types=TechnicalParameters.bars): pass
class StatisticFile(TechnicalFile, variable=Variables.Technicals.STATISTIC, types=TechnicalParameters.statistic): pass
class StochasticFile(TechnicalFile, variable=Variables.Technicals.STOCHASTIC, types=TechnicalParameters.stochastic): pass


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


class TechnicalCalculator(object):
    def __init__(self, *args, technical, **kwargs):
        self.__calculation = TechnicalCalculation[technical](*args, **kwargs)
        self.__variables = TechnicalVariables(*args, technical=technical, **kwargs)

    def __call__(self, bars, *args, **kwargs):
        assert isinstance(bars, pd.DataFrame)
        for symbol, dataframe in self.symbols(bars):
            technicals = self.execute(dataframe, *args, **kwargs)
            size = len(bars.dropna(how="all", inplace=False).index)
            string = f"Calculated: {repr(self)}|{str(symbol)}[{size:.0f}]"
            self.logger.info(string)
            if bool(technicals.empty): continue
            yield technicals

    def symbols(self, bars):
        assert isinstance(bars, pd.DataFrame)
        for (ticker,), dataframe in bars.groupby(self.variables.symbol):
            if bool(dataframe.empty): continue
            symbol = Symbol(ticker)
            yield symbol, dataframe

    def execute(self, bars, *args, **kwargs):
        assert isinstance(bars, pd.DataFrame)
        technicals = self.calculate(bars, *args, **kwargs)
        return technicals

    def calculate(self, bars, *args, **kwargs):
        assert isinstance(bars, pd.DataFrame)
        bars = bars.sort_values("date", ascending=False, inplace=False)
        technicals = self.calculation(bars, *args, **kwargs)
        assert isinstance(technicals, pd.DataFrame)
        technicals = technicals if bool(technicals) else pd.DataFrame(columns=self.variables.header)
        return technicals

    @property
    def calculation(self): return self.__calculations
    @property
    def variables(self): return self.__variables



