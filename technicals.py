# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 2024
@name:   Technical Objects
@author: Jack Kirby Cook

"""

import types
import logging
import numpy as np
import pandas as pd
from abc import ABC

from support.calculations import Calculation, Equation, Variable
from support.mixins import Emptying, Sizing, Partition
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = []
__copyright__ = "Copyright 2024, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


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


class TechnicalCalculator(Sizing, Emptying, Partition):
    def __init__(self, *args, technicals, **kwargs):
        assert all([technical in list(Technicals) for technical in technicals])
        super().__init__(*args, **kwargs)
        technicals = list(dict(TechnicalCalculation).keys()) if not bool(technicals) else list(technicals)
        calculations = dict(TechnicalCalculation).items()
        calculations = {(STOCK, technical): calculation(*args, **kwargs) for technical, calculation in calculations if technical in technicals}
        self.calculations = calculations

    def execute(self, bars, *args, **kwargs):
        assert isinstance(bars, pd.DataFrame)
        if self.empty(bars): return
        for partition, dataframe in self.partition(bars, by=Symbol):
            contents = {dataset: self.calculation(dataframe, *args, **kwargs) for dataset, calculation in self.calculations.items()}
            for dataset, content in contents.items():
                string = "|".join(list(map(str, dataset)))
                size = self.size(dataframe)
                string = f"Downloaded: {repr(self)}|{str(string)}|{str(partition)}[{int(size):.0f}]"
                __logger__.info(string)
            if self.empty(contents): continue
            yield contents



