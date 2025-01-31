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

from finance.variables import Variables, Querys
from support.calculations import Calculation, Equation, Variable
from support.files import Directory, Loader, Saver, File
from support.mixins import Emptying, Sizing, Partition
from support.meta import RegistryMeta, MappingMeta
from support.variables import Category

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TechnicalCalculator", "TechnicalDirectory", "TechnicalLoader", "TechnicalSaver", "TechnicalFiles"]
__copyright__ = "Copyright 2024, Jack Kirby Cook"
__license__ = "MIT License"


class TechnicalParameters(metaclass=MappingMeta):
    types = {"ticker": str, "price underlying strike bid ask open close high low": np.float32, "trend volatility oscillator": np.float32, "volume": np.int64}
    types = {key: value for keys, value in types.items() for key in str(key).split(" ")}
    parsers = dict(instrument=Variables.Securities.Instrument, option=Variables.Securities.Option, position=Variables.Securities.Position)
    formatters = dict(instrument=int, option=int, position=int)
    dates = dict(date="Y%m%d", expire="Y%m%d", current="%Y%m%d-%H%M")

class StockTradeFile(File, order=["ticker", "current", "price"], **dict(TechnicalParameters)): pass
class StockQuoteFile(File, order=["ticker", "current", "bid", "ask", "demand", "supply"], **dict(TechnicalParameters)): pass
class StockBarsFile(File, order=["ticker", "date", "open", "close", "high", "low", "price"], **dict(TechnicalParameters)): pass
class StockStatisticFile(File, order=["ticker", "date", "price", "trend", "volatility"], **dict(TechnicalParameters)): pass
class StockStochasticFile(File, order=["ticker", "date", "price", "oscillator"], **dict(TechnicalParameters)): pass
class OptionTradeFile(File, order=["ticker", "expire", "strike", "option", "current", "price", "underlying"], **dict(TechnicalParameters)): pass
class OptionQuoteFile(File, order=["ticker", "expire", "strike", "option", "current", "bid", "ask", "demand", "supply", "underlying"], **dict(TechnicalParameters)): pass

class TechnicalFiles(Category):
    class Stocks(Category): Trade, Quote, Bars, Statistic, Stochastic = StockTradeFile, StockQuoteFile, StockBarsFile, StockStatisticFile, StockStochasticFile
    class Options(Category): Trade, Quote = OptionTradeFile, OptionQuoteFile

class TechnicalDirectory(Directory, query=Querys.Symbol): pass
class TechnicalLoader(Loader, query=Querys.Symbol): pass
class TechnicalSaver(Saver, query=Querys.Symbol): pass


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
class StatisticCalculation(TechnicalCalculation, equation=StatisticEquation):
    def execute(self, bars, *args, period, **kwargs):
        assert (bars["ticker"].to_numpy()[0] == bars["ticker"]).all()
        with self.equation(bars, period=period) as equation:
            yield bars["ticker"]
            yield bars["date"]
            yield bars["price"]
            yield equation.m()
            yield equation.δ()

class StochasticCalculation(TechnicalCalculation, equation=StochasticEquation):
    def execute(self, bars, *args, period, **kwargs):
        assert (bars["ticker"].to_numpy()[0] == bars["ticker"]).all()
        bars = bars.sort_values("date", ascending=True, inplace=False)
        with self.equation(bars, period=period) as equation:
            yield bars["ticker"]
            yield bars["date"]
            yield bars["price"]
            yield equation.xk()


class TechnicalCalculator(Sizing, Emptying, Partition, query=Querys.Symbol, title="Calculated"):
    def __init__(self, *args, technicals, **kwargs):
        assert all([technical in list(Variables.Analysis.Technical) for technical in technicals])
        super().__init__(*args, **kwargs)
        technicals = list(dict(TechnicalCalculation).keys()) if not bool(technicals) else list(technicals)
        calculations = {technical: calculation(*args, **kwargs) for technical, calculation in dict(TechnicalCalculation).items() if technical in technicals}
        self.__calculations = calculations

    def execute(self, bars, *args, **kwargs):
        assert isinstance(bars, pd.DataFrame)
        if self.empty(bars): return
        for symbol, dataframe in self.partition(bars):
            technicals = self.calculate(dataframe, *args, **kwargs)
            size = self.size(technicals)
            string = f"{str(symbol)}[{int(size):.0f}]"
            self.console(string)
            if self.empty(technicals): continue
            yield technicals

    def calculate(self, bars, *args, **kwargs):
        for technical, calculation in self.calculations.items():
            technicals = calculation(bars, *args, **kwargs)
            bars = bars.merge(technicals, how="outer", on=list(Querys.History), sort=False)
        return bars

    @property
    def calculations(self): return self.__calculations



