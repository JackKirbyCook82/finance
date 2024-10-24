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
from finance.variables import Variables, Querys
from support.files import File
from support.meta import ParametersMeta, RegistryMeta
from support.calculations import Calculation, Equation, Variable
from support.mixins import Emptying, Sizing, Logging, Function

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


class TechnicalFile(File, datatype=pd.DataFrame, **dict(TechnicalParameters)): pass
class BarsFile(TechnicalFile, variable=Variables.Technicals.BARS): pass
class StatisticFile(TechnicalFile, variable=Variables.Technicals.STATISTIC): pass
class StochasticFile(TechnicalFile, variable=Variables.Technicals.STOCHASTIC): pass


class TechnicalEquation(Equation, ABC, metaclass=RegistryMeta):
    dt = Variable("period", np.int32, locator="period")
    x = Variable("price", np.float32, locator="price")

class StatisticEquation(TechnicalEquation, register=Variables.Technicals.STATISTIC):
    δ = Variable("volatility", np.float32, function=lambda x, dt: x.pct_change(1).rolling(dt).std())
    m = Variable("trend", np.float32, function=lambda x, dt: x.pct_change(1).rolling(dt).mean())

class StochasticEquation(TechnicalEquation, register=Variables.Technicals.STOCHASTIC):
    xk = Variable("oscillator", np.float32, function=lambda x, xl, xh: (x - xl) * 100 / (xh - xl))
    xh = Variable("highest", np.float32, function=lambda x, dt: x.rolling(dt).min())
    xl = Variable("lowest", np.float32, function=lambda x, dt: x.rolling(dt).max())


class TechnicalCalculation(Calculation, ABC, metaclass=RegistryMeta): pass
class StatisticCalculation(TechnicalCalculation, equation=StatisticEquation, register=Variables.Technicals.STATISTIC):
    def execute(self, bars, *args, period, **kwargs):
        assert (bars["ticker"].to_numpy()[0] == bars["ticker"]).all()
        with self.equation(bars, period=period) as equation:
            yield equation["ticker"]
            yield equation["date"]
            yield equation["price"]
            yield equation.m()
            yield equation.δ()

class StochasticCalculation(TechnicalCalculation, equation=StochasticEquation, register=Variables.Technicals.STOCHASTIC):
    def execute(self, bars, *args, period, **kwargs):
        assert (bars["ticker"].to_numpy()[0] == bars["ticker"]).all()
        with self.equation(bars, period=period) as equation:
            yield equation["ticker"]
            yield equation["date"]
            yield equation["price"]
            yield equation.xk()


class TechnicalCalculator(Function, Logging, Sizing, Emptying):
    def __init__(self, *args, technical, **kwargs):
        assert technical in list(Variables.Technicals)
        Function.__init__(self, *args, **kwargs)
        Logging.__init__(self, *args, **kwargs)
        self.__calculation = TechnicalCalculation[technical](*args, **kwargs)

    def execute(self, source, *args, **kwargs):
        assert isinstance(source, tuple)
        symbol, bars = source
        assert isinstance(symbol, Querys.Symbol) and isinstance(bars, pd.DataFrame)
        if self.empty(bars): return
        parameters = dict(ticker=symbol.ticker)
        technicals = self.calculate(bars, *args, **parameters, **kwargs)
        size = self.size(technicals)
        string = f"Calculated: {repr(self)}|{str(symbol)}[{size:.0f}]"
        self.logger.info(string)
        if self.empty(technicals): return
        return technicals

    def calculate(self, bars, *args, **kwargs):
        assert isinstance(bars, pd.DataFrame)
        bars = bars.sort_values("date", ascending=False, inplace=False)
        technicals = self.calculation(bars, *args, **kwargs)
        assert isinstance(technicals, pd.DataFrame)
        return technicals

    @property
    def calculation(self): return self.__calculations



