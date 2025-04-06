# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 2024
@name:   Technical Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from abc import ABC

from finance.variables import Variables, Querys
from support.calculations import Calculation, Equation, Variable
from support.mixins import Emptying, Sizing, Partition, Logging
from support.meta import RegistryMeta, ParameterMeta
from support.variables import Category
from support.files import File

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TechnicalCalculator", "TechnicalFiles"]
__copyright__ = "Copyright 2024, Jack Kirby Cook"
__license__ = "MIT License"


class TechnicalParameters(metaclass=ParameterMeta):
    types = {"ticker": str, "open close high low": np.float32, "price trend volatility": np.float32}
    parsers = dict(instrument=Variables.Securities.Instrument, option=Variables.Securities.Option, position=Variables.Securities.Position)
    formatters = dict(instrument=int, option=int, position=int)
    dates = dict(date="%Y%m%d", expire="%Y%m%d")

class TechnicalFile(File, **dict(TechnicalParameters)): pass
class StockBarsFile(TechnicalFile, order=["ticker", "date", "open", "close", "high", "low", "price"]): pass
class StockStatisticFile(TechnicalFile, order=["ticker", "date", "price", "trend", "volatility"]): pass
class StockStochasticFile(TechnicalFile, order=["ticker", "date", "price", "oscillator"]): pass

class TechnicalFiles(Category):
    class Stocks(Category): Bars, Statistic, Statistic = StockBarsFile, StockStatisticFile, StockStochasticFile


class TechnicalEquation(Equation, ABC, datatype=pd.Series, vectorize=False):
    x = Variable.Independent("x", "price", np.float32, locator="price")
    dt = Variable.Constant("dt", "period", np.int32, locator="period")

class StatisticEquation(TechnicalEquation):
    δ = Variable.Dependent("δ", "volatility", np.float32, function=lambda x, *, dt: x.pct_change(1).rolling(dt).std())
    m = Variable.Dependent("m", "trend", np.float32, function=lambda x, *, dt: x.pct_change(1).rolling(dt).mean())

class StochasticEquation(TechnicalEquation):
    xk = Variable.Dependent("xk", "oscillator", np.float32, function=lambda x, xl, xh: (x - xl) * 100 / (xh - xl))
    xh = Variable.Dependent("xh", "highest", np.float32, function=lambda x, *, dt: x.rolling(dt).min())
    xl = Variable.Dependent("xl", "lowest", np.float32, function=lambda x, *, dt: x.rolling(dt).max())


class TechnicalCalculation(Calculation, ABC, metaclass=RegistryMeta): pass
class StatisticCalculation(TechnicalCalculation, equation=StatisticEquation, register=Variables.Analysis.Technical.STOCHASTIC):
    def execute(self, bars, *args, period, **kwargs):
        assert (bars["ticker"].to_numpy()[0] == bars["ticker"]).all()
        with self.equation(bars, period=period) as equation:
            yield equation.m()
            yield equation.δ()

class StochasticCalculation(TechnicalCalculation, equation=StochasticEquation, register=Variables.Analysis.Technical.STATISTIC):
    def execute(self, bars, *args, period, **kwargs):
        assert (bars["ticker"].to_numpy()[0] == bars["ticker"]).all()
        with self.equation(bars, period=period) as equation:
            yield equation.xk()


class TechnicalCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, technicals, **kwargs):
        assert all([technical in list(Variables.Analysis.Technical) for technical in technicals])
        super().__init__(*args, **kwargs)
        technicals = list(dict(TechnicalCalculation).keys()) if not bool(technicals) else list(technicals)
        calculations = {technical: calculation(*args, **kwargs) for technical, calculation in dict(TechnicalCalculation).items() if technical in technicals}
        self.__calculations = calculations

    def execute(self, bars, *args, **kwargs):
        assert isinstance(bars, pd.DataFrame)
        if self.empty(bars): return
        technicals = self.calculate(bars, *args, **kwargs)
        symbols = self.groups(technicals, by=Querys.Symbol)
        symbols = ",".join(list(map(str, symbols)))
        size = self.size(technicals)
        self.console(f"{str(symbols)}[{int(size):.0f}]")
        if self.empty(technicals): return
        yield technicals

    def calculate(self, bars, *args, **kwargs):
        technicals = list(self.calculator(bars, *args, **kwargs))
        technicals = pd.concat(technicals, axis=0)
        return technicals

    def calculator(self, bars, *args, **kwargs):
        for symbol, dataframe in self.partition(bars, by=Querys.Symbol):
            technicals = list(self.technicals(dataframe, *args, **kwargs))
            technicals = pd.concat([bars] + technicals, axis=1)
            yield technicals

    def technicals(self, bars, *args, **kwargs):
        assert (bars["ticker"].to_numpy()[0] == bars["ticker"]).all()
        bars = bars.sort_values("date", ascending=True, inplace=False)
        for technical, calculation in self.calculations.items():
            technicals = calculation(bars, *args, **kwargs)
            yield technicals

    @property
    def calculations(self): return self.__calculations



