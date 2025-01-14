# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 2024
@name:   Technical Objects
@author: Jack Kirby Cook

"""

import types
import numpy as np
import pandas as pd
from abc import ABC

from finance.variables import Variables
from support.mixins import Emptying, Sizing, Logging, Segregating
from support.calculations import Calculation, Equation, Variable
from support.meta import RegistryMeta, MappingMeta
from support.files import File

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StatisticCalculator", "StochasticCalculator", "HistoryFile", "StatisticFile", "StochasticFile"]
__copyright__ = "Copyright 2024, Jack Kirby Cook"
__license__ = "MIT License"


class TechnicalParameters(metaclass=MappingMeta):
    order = ["ticker", "date", "price", "open", "close", "high", "low", "trend", "volatility", "oscillator"]
    types = {"ticker": str, "volume": np.int64} | {column: np.float32 for column in ("price", "open", "close", "high", "low")}
    types.update({"trend": np.float32, "volatility": np.float32, "oscillator": np.float32})
    dates = {"date": "%Y%m%d"}

class TechnicalFile(File, ABC, **dict(TechnicalParameters)): pass
class HistoryFile(TechnicalFile, variable=Variables.Technicals.HISTORY): pass
class StatisticFile(TechnicalFile, variable=Variables.Technicals.STATISTIC): pass
class StochasticFile(TechnicalFile, variable=Variables.Technicals.STOCHASTIC): pass


class TechnicalEquation(Equation, ABC):
    dt = Variable("dt", "period", np.int32, types.NoneType, locator="period")
    x = Variable("x", "price", np.float32, pd.Series, locator="price")

class StatisticEquation(TechnicalEquation):
    δ = Variable("δ", "volatility", np.float32, pd.Series, vectorize=False, function=lambda x, dt: x.pct_change(1).rolling(dt).std())
    m = Variable("m", "trend", np.float32, pd.Series, vectorize=False, function=lambda x, dt: x.pct_change(1).rolling(dt).mean())

class StochasticEquation(TechnicalEquation):
    xk = Variable("xk", "oscillator", np.float32, pd.Series, vectorize=False, function=lambda x, xl, xh: (x - xl) * 100 / (xh - xl))
    xh = Variable("xh", "highest", np.float32, pd.Series, vectorize=False, function=lambda x, dt: x.rolling(dt).min())
    xl = Variable("xl", "lowest", np.float32, pd.Series, vectorize=False, function=lambda x, dt: x.rolling(dt).max())


class TechnicalCalculation(Calculation, ABC, metaclass=RegistryMeta): pass
class StatisticCalculation(TechnicalCalculation, equation=StatisticEquation, register=Variables.Technicals.STATISTIC):
    def execute(self, history, *args, period, **kwargs):
        assert (history["ticker"].to_numpy()[0] == history["ticker"]).all()
        with self.equation(history, period=period) as equation:
            yield history["ticker"]
            yield history["date"]
            yield history["price"]
            yield equation.m()
            yield equation.δ()

class StochasticCalculation(TechnicalCalculation, equation=StochasticEquation, register=Variables.Technicals.STOCHASTIC):
    def execute(self, history, *args, period, **kwargs):
        assert (history["ticker"].to_numpy()[0] == history["ticker"]).all()
        history = history.sort_values("date", ascending=True, inplace=False)
        with self.equation(history, period=period) as equation:
            yield history["ticker"]
            yield history["date"]
            yield history["price"]
            yield equation.xk()


class TechnicalCalculator(Segregating, Sizing, Emptying, Logging):
    def __init__(self, *args, technical, **kwargs):
        assert technical in list(Variables.Technicals)
        super().__init__(*args, **kwargs)
        self.__calculation = TechnicalCalculation[technical](*args, **kwargs)
        self.__technical = technical

    def execute(self, history, *args, **kwargs):
        assert isinstance(history, pd.DataFrame)
        if self.empty(history): return
        for query, dataframe in self.segregate(history, *args, **kwargs):
            technicals = self.calculate(dataframe, *args, **kwargs)
            size = self.size(technicals)
            string = f"Calculated: {repr(self)}|{str(query)}[{int(size):.0f}]"
            self.logger.info(string)
            if self.empty(technicals): continue
            yield technicals

    def calculate(self, history, *args, **kwargs):
        assert isinstance(history, pd.DataFrame)
        technicals = self.calculation(history, *args, **kwargs)
        assert isinstance(technicals, pd.DataFrame)
        return technicals

    @property
    def calculation(self): return self.__calculation
    @property
    def technical(self): return self.__technical


class StatisticCalculator(TechnicalCalculator):
    def __init__(self, *args, **kwargs):
        parameters = dict(technical=Variables.Technicals.STATISTIC)
        super().__init__(*args, **parameters, **kwargs)


class StochasticCalculator(TechnicalCalculator):
    def __init__(self, *args, **kwargs):
        parameters = dict(technical=Variables.Technicals.STOCHASTIC)
        super().__init__(*args, **parameters, **kwargs)



