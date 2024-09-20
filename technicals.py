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
from support.processes import Calculator
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TechnicalCalculator"]
__copyright__ = "Copyright 2024, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


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


class TechnicalVariables(object):
    data = {Variables.Technicals.STATISTIC: ["price", "trend", "volatility"], Variables.Technicals.STOCHASTIC: ["price", "oscillator"]}
    axes = {Variables.Querys.HISTORY: ["date", "ticker"]}

    def __init__(self, *args, technical, **kwargs):
        assert technical in list(Variables.Technicals)
        index = self.axes[Variables.Querys.HISTORY]
        columns = self.data[technical]
        self.header = list(index) + list(columns)
        self.technical = technical
        

class TechnicalCalculator(Calculator, calculations=dict(TechnicalCalculation), variables=TechnicalVariables):
    def execute(self, symbol, bars, *args, **kwargs):
        assert isinstance(symbol, Symbol) and isinstance(bars, pd.DataFrame)
        technicals = self.technicals(bars, *args, **kwargs)
        technicals = technicals if bool(technicals) else pd.DataFrame(columns=self.variables.header)
        size = self.size(technicals)
        string = f"{str(self.title)}: {repr(self)}|{str(symbol)}[{size:.0f}]"
        self.logger.info(string)
        return technicals

    def technicals(self, bars, *args, **kwargs):
        if bool(bars.empty): return
        technicals = self.calculations[self.variables.technical](bars, *args, **kwargs)
        return technicals




