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

from finance.variables import Variables
from support.calculations import Variable, Equation, Calculation
from support.pipelines import Processor
from support.meta import ParametersMeta
from support.files import File

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TechnicalFiles", "TechnicalCalculator"]
__copyright__ = "Copyright 2024, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class TechnicalParameters(object, metaclass=ParametersMeta):
    bars = {"ticker": str, "volume": np.int64} | {column: np.float32 for column in ("price", "open", "close", "high", "low")}
    stochastic = {"trend": np.float32, "volatility": np.float32}
    statistic = {"oscillator": np.float32}
    types = bars | statistic | stochastic
    dates = {"date": "%Y%m%d"}
    filename = lambda symbol: str(symbol.ticker).upper()
    datatype = pd.DataFrame


class BarsFile(File, variable=Variables.Technicals.BARS, **dict(TechnicalParameters)): pass
class StatisticFile(File, variable=Variables.Technicals.STATISTIC, **dict(TechnicalParameters)): pass
class StochasticFile(File, variable=Variables.Technicals.STOCHASTIC, **dict(TechnicalParameters)): pass
class TechnicalFiles(object): Bars = BarsFile; Statistic = StatisticFile; Stochastic = StochasticFile


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


class TechnicalCalculator(Processor, title="Calculated", reporting=True, variable=Variables.Querys.SYMBOL):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        calculations = {variables["technical"]: calculation for variables, calculation in ODict(list(TechnicalCalculation)).items()}
        self.__calculations = {variable: calculation(*args, **kwargs) for variable, calculation in calculations.items()}

    def processor(self, contents, *args, **kwargs):
        bars = contents[Variables.Technicals.BARS]
        assert isinstance(bars, pd.DataFrame)
        technicals = list(self.calculate(bars, *args, **kwargs))
        if not bool(technicals): return
        yield contents | ODict(technicals)

    def calculate(self, bars, *args, **kwargs):
        for variable, calculation in self.calculations.items():
            technical = calculation(bars, *args, **kwargs)
            if bool(technical.empty): continue
            yield variable, technical

    @property
    def calculations(self): return self.__calculations



