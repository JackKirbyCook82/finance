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

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TechnicalCalculator"]
__copyright__ = "Copyright 2024, Jack Kirby Cook"
__license__ = "MIT License"


class TechnicalEquation(Equation, ABC, datatype=pd.Series, vectorize=False):
    x = Variable.Independent("x", "price", np.float32, locator="price")
    dt = Variable.Constant("dt", "period", np.int32, locator="period")


class StatisticEquation(TechnicalEquation, register=Variables.Technical.STATISTIC):
    δ = Variable.Dependent("δ", "volatility", np.float32, function=lambda x, *, dt: x.pct_change(1).rolling(dt).std())
    m = Variable.Dependent("m", "trend", np.float32, function=lambda x, *, dt: x.pct_change(1).rolling(dt).mean())

    def execute(self, *args, **kwargs):
        yield self.m()
        yield self.δ()


class StochasticEquation(TechnicalEquation, register=Variables.Technical.STOCHASTIC):
    xk = Variable.Dependent("xk", "oscillator", np.float32, function=lambda x, xl, xh: (x - xl) * 100 / (xh - xl))
    xh = Variable.Dependent("xh", "highest", np.float32, function=lambda x, *, dt: x.rolling(dt).min())
    xl = Variable.Dependent("xl", "lowest", np.float32, function=lambda x, *, dt: x.rolling(dt).max())

    def execute(self, *args, **kwargs):
        yield self.xk()


class TechnicalCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, technicals, **kwargs):
        assert all([technical in list(Variables.Technical) for technical in technicals])
        super().__init__(*args, **kwargs)
        equations = [TechnicalEquation[technical] for technical in technicals]
        self.__calculation = Calculation[pd.Series](*args, equations=equations, **kwargs)

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
        header = ["ticker", "date", "price"]
        for dataframe in bars:
            assert (dataframe["ticker"].to_numpy()[0] == dataframe["ticker"]).all()
            dataframe = dataframe.sort_values("date", ascending=True, inplace=False)
            technicals = self.calculation(dataframe, *args, **kwargs)
            assert isinstance(technicals, pd.DataFrame)
            results = pd.concat([dataframe[header], technicals], axis=1)
            yield results

    @property
    def calculation(self): return self.__calculation



