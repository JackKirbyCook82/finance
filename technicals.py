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
from collections import OrderedDict as ODict

from finance.variables import Variables, Symbol
from support.calculations import Variable, Equation, Calculation
from support.meta import ParametersMeta
from support.mixins import Sizing

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


class TechnicalCalculation(Calculation, ABC, fields=["technical"]): pass
class StatisticCalculation(TechnicalCalculation, technical=Variables.Technicals.STATISTIC):
    @staticmethod
    def execute(bars, *args, period, **kwargs):
        assert (bars["ticker"].to_numpy()[0] == bars["ticker"]).all()
        yield from iter([bars["ticker"], bars["date"], bars["price"]])
        yield bars["price"].pct_change(1).rolling(period).mean().rename("trend")
        yield bars["price"].pct_change(1).rolling(period).std().rename("volatility")

class StochasticCalculation(TechnicalCalculation, technical=Variables.Technicals.STOCHASTIC, equation=StochasticEquation):
    def execute(self, bars, *args, period, **kwargs):
        assert (bars["ticker"].to_numpy()[0] == bars["ticker"]).all()
        equation = self.equation(*args, **kwargs)
        lowest = bars["low"].rolling(period).min().rename("lowest")
        highest = bars["high"].rolling(period).max().rename("highest")
        bars = pd.concat([bars, lowest, highest], axis=1)
        yield from iter([bars["ticker"], bars["date"], bars["price"]])
        yield equation.xk(bars)


class TechnicalAxes(object, metaclass=ParametersMeta):
    bars = ["price", "open", "close", "high", "low", "volume"]
    statistic = ["price", "trend", "volatility"]
    stochastic = ["price", "oscillator"]
    index = ["ticker", "date"]

    def __init__(self, *args, technical, **kwargs):
        technical = str(technical).lower()
        self.header = self.index + list(getattr(self, technical))
        self.technical = str(technical).lower()


class TechnicalCalculator(Sizing):
    def __init__(self, *args, technical, **kwargs):
        super().__init__(*args, **kwargs)
        calculations = {variables["technical"]: calculation for variables, calculation in ODict(list(TechnicalCalculation)).items()}
        self.__axes = TechnicalAxes(*args, technical=technical, **kwargs)
        self.__calculation = calculations[technical](*args, **kwargs)
        self.__logger = __logger__

    def calculate(self, symbol, bars, *args, **kwargs):
        assert isinstance(symbol, Symbol) and isinstance(bars, pd.DataFrame)
        technicals = self.technicals(bars, *args, **kwargs)
        technicals = technicals if bool(technicals) else pd.DataFrame(columns=self.axes.header)
        size = self.size(technicals)
        string = f"Calculated: {repr(self)}|{str(symbol)}[{size:.0f}]"
        self.logger.info(string)
        return technicals

    def technicals(self, bars, *args, **kwargs):
        if bool(bars.empty): return
        technicals = self.calculation(bars, *args, **kwargs)
        return technicals

    @property
    def calculation(self): return self.__calculation
    @property
    def logger(self): return self.__logger



