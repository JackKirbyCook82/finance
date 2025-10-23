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

from finance.concepts import Concepts, Querys
from calculations import Equation, Variables, Algorithms, Computations
from support.mixins import Emptying, Sizing, Partition, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TechnicalCalculator"]
__copyright__ = "Copyright 2024, Jack Kirby Cook"
__license__ = "MIT License"


class TechnicalEquation(ABC):
    x = Variables.Independent("x", "adjusted", np.float32, locator="adjusted")
    s = Variables.Independent("s", "ticker", Date, locator="ticker")
    t = Variables.Independent("t", "date", Date, locator="date")
    dt = Variables.Constant("dt", "period", np.int32, locator="period")

    def execute(self, bars, /, period):
        parameters = dict(period=period)
        yield from super().execute(bars, **parameters)
        yield self.s(bars, **parameters)
        yield self.x(bars, **parameters)
        yield self.t(bars, **parameters)

class BarsEquation(TechnicalEquation, register=Concepts.Technical.BARS):
    xo = Variables.Independent("xo", "open", np.float32, locator="open")
    xc = Variables.Independent("xc", "close", np.float32, locator="close")
    xl = Variables.Independent("xl", "low", np.float32, locator="low")
    xh = Variables.Independent("xh", "high", np.float32, locator="high")

    def execute(self, bars, /, period):
        parameters = dict(period=period)
        yield from super().execute(bars, **parameters)
        yield self.xo(bars, **parameters)
        yield self.xc(bars, **parameters)
        yield self.xl(bars, **parameters)
        yield self.xh(bars, **parameters)

class StatisticEquation(TechnicalEquation, register=Concepts.Technical.STATISTIC):
    δ = Variables.Dependent("δ", "volatility", np.float32, function=lambda x, *, dt: x.pct_change(1).rolling(dt).std())
    μ = Variables.Dependent("μ", "trend", np.float32, function=lambda x, *, dt: x.pct_change(1).rolling(dt).mean())

    def execute(self, bars, /, period):
        parameters = dict(period=period)
        yield from super().execute(bars, **parameters)
        yield self.δ(bars, **parameters)
        yield self.μ(bars, **parameters)

class StochasticEquation(TechnicalEquation, register=Concepts.Technical.STOCHASTIC):
    xk = Variables.Dependent("xk", "oscillator", np.float32, function=lambda x, xkl, xkh: (x - xkl) * 100 / (xkh - xkl))
    xkh = Variables.Dependent("xkh", "highest", np.float32, function=lambda x, *, dt: x.rolling(dt).min())
    xkl = Variables.Dependent("xkl", "lowest", np.float32, function=lambda x, *, dt: x.rolling(dt).max())

    def execute(self, bars, /, period):
        parameters = dict(period=period)
        yield from super().execute(bars, **parameters)
        yield self.xk(bars, **parameters)


class TechnicalCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, technicals, **kwargs):
        assert all([technical in list(Concepts.Technical) for technical in technicals])
        super().__init__(*args, **kwargs)
        equations = [equation for technical, equation in iter(TechnicalEquation) if technical in technicals]
        self.__equation = (TechnicalEquation + equations)(*args, **kwargs)

    def execute(self, bars, /, **kwargs):
        assert isinstance(bars, pd.DataFrame)
        if self.empty(bars): return
        symbols = self.keys(bars, by=Querys.Symbol)
        symbols = ",".join(list(map(str, symbols)))
        technicals = self.calculate(bars, **kwargs)
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

    def calculator(self, bars, *args, period, **kwargs):
        assert isinstance(bars, list) and all([isinstance(dataframe, pd.DataFrame) for dataframe in bars])
        for dataframe in bars:
            assert (dataframe["ticker"].to_numpy()[0] == dataframe["ticker"]).all()
            dataframe = dataframe.sort_values("date", ascending=True, inplace=False)
            technicals = self.equation(dataframe, period=period)
            assert isinstance(technicals, pd.DataFrame)
            yield technicals

    @property
    def equation(self): return self.__equation



