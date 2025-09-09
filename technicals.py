# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 2024
@name:   Technical Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from abc import ABC
from datetime import date as Date

import calculations as calc
from finance.concepts import Concepts, Querys
from support.mixins import Emptying, Sizing, Partition, Logging
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TechnicalCalculator"]
__copyright__ = "Copyright 2024, Jack Kirby Cook"
__license__ = "MIT License"


class TechnicalEquationMeta(RegistryMeta, type(calc.Equations.Table)): pass
class TechnicalEquation(calc.Equations.Table, ABC, metaclass=TechnicalEquationMeta):
    x = calc.Variables.Independent("x", "adjusted", np.float32, locator="adjusted")
    s = calc.Variables.Independent("s", "ticker", Date, locator="ticker")
    t = calc.Variables.Independent("t", "date", Date, locator="date")
    dt = calc.Variables.Constant("dt", "period", np.int32, locator="period")

#    def execute(self, *args, **kwargs):
#        yield from super().execute(*args, **kwargs)
#        yield self.x()
#        yield self.s()
#        yield self.t()


class BarsEquation(TechnicalEquation, signature="[xo,xc,xl,xh]->[xo,xc,xl,xh]", register=Concepts.Technical.BARS):
    xo = calc.Variables.Independent("xo", "open", np.float32, locator="open")
    xc = calc.Variables.Independent("xc", "close", np.float32, locator="close")
    xl = calc.Variables.Independent("xc", "low", np.float32, locator="low")
    xh = calc.Variables.Independent("xc", "high", np.float32, locator="high")

#    def execute(self, *args, **kwargs):
#        yield from super().execute(*args, **kwargs)
#        yield self.xo()
#        yield self.xc()
#        yield self.xh()
#        yield self.xl()


class StatisticEquation(TechnicalEquation, signature="[x,dt]->[μ,δ]", register=Concepts.Technical.STATISTIC):
    δ = calc.Variables.Dependent("δ", "volatility", np.float32, function=lambda x, *, dt: x.pct_change(1).rolling(dt).std())
    μ = calc.Variables.Dependent("μ", "trend", np.float32, function=lambda x, *, dt: x.pct_change(1).rolling(dt).mean())

#    def execute(self, *args, **kwargs):
#        yield from super().execute(*args, **kwargs)
#        yield self.μ()
#        yield self.δ()


class StochasticEquation(TechnicalEquation, signature="[x,dt]->[xk]", register=Concepts.Technical.STOCHASTIC):
    xk = calc.Variables.Dependent("xk", "oscillator", np.float32, function=lambda x, xkl, xkh: (x - xkl) * 100 / (xkh - xkl))
    xkh = calc.Variables.Dependent("xkh", "highest", np.float32, function=lambda x, *, dt: x.rolling(dt).min())
    xkl = calc.Variables.Dependent("xkl", "lowest", np.float32, function=lambda x, *, dt: x.rolling(dt).max())

#    def execute(self, *args, **kwargs):
#        yield from super().execute(*args, **kwargs)
#        yield self.xk()


class TechnicalCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, technicals, **kwargs):
        assert all([technical in list(Concepts.Technical) for technical in technicals])
        super().__init__(*args, **kwargs)
        equations = [equation for technical, equation in iter(TechnicalEquation) if technical in technicals]
        self.__calculation = Calculation[pd.Series](*args, required=equations, **kwargs)

    def execute(self, bars, *args, **kwargs):
        assert isinstance(bars, pd.DataFrame)
        if self.empty(bars): return
        symbols = self.keys(bars, by=Querys.Symbol)
        symbols = ",".join(list(map(str, symbols)))
        technicals = self.calculate(bars, *args, **kwargs)
        size = self.size(technicals)
        self.console(f"{str(symbols)}[{int(size):.0f}]")
        if self.empty(technicals): return
        yield technicals

    def calculate(self, bars, *args, **kwargs):
        assert isinstance(bars, pd.DataFrame)
        bars = list(self.values(bars, by=Querys.Symbol))
        technicals = list(self.calculator(bars, *args, **kwargs))
        technicals = pd.concat(technicals, axis=0)
        technicals = technicals.reset_index(drop=True, inplace=False)
        return technicals

    def calculator(self, bars, *args, **kwargs):
        assert isinstance(bars, list) and all([isinstance(dataframe, pd.DataFrame) for dataframe in bars])
        for dataframe in bars:
            assert (dataframe["ticker"].to_numpy()[0] == dataframe["ticker"]).all()
            dataframe = dataframe.sort_values("date", ascending=True, inplace=False)
            technicals = self.calculation(dataframe, *args, **kwargs)
            assert isinstance(technicals, pd.DataFrame)
            yield technicals

    @property
    def calculation(self): return self.__calculation



