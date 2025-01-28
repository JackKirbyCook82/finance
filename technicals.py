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
from support.mixins import Emptying, Sizing, Partition
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TechnicalCalculator"]
__copyright__ = "Copyright 2024, Jack Kirby Cook"
__license__ = "MIT License"


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



