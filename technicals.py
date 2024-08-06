# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 2024
@name:   Technical Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from abc import ABC
from collections import OrderedDict as ODict

from finance.variables import Variables
from support.calculations import Variable, Equation, Calculation
from support.pipelines import Processor
from support.files import File

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TechnicalFiles", "TechnicalCalculator"]
__copyright__ = "Copyright 2024, Jack Kirby Cook"
__license__ = "MIT License"


technical_dates = {"date": "%Y%m%d"}
technical_types = {"ticker": str, "high": np.float32, "low": np.float32, "open": np.float32, "close": np.float32, "price": np.float32, "volume": np.float32, "trend": np.float32, "volatility": np.float32, "oscillator": np.float32}
technical_filename = lambda query: str(query.ticker).upper()
technical_formatter = lambda self, *, results, elapsed, **kw: f"{str(self.title)}: {repr(self)}|{str(results[Variables.Querys.SYMBOL])}[{elapsed:.02f}s]"
technical_parameters = dict(datatype=pd.DataFrame, filename=technical_filename, dates=technical_dates, types=technical_types)
bars_header = ["date", "ticker", "high", "low", "open", "close", "price", "volume"]
statistic_header = ["date", "ticker", "price", "trend", "volatility"]
stochastic_header = ["date", "ticker", "price", "oscillator"]


class BarsFile(File, variable=Variables.Technicals.BARS, header=bars_header, **technical_parameters): pass
class StatisticFile(File, variable=Variables.Technicals.STATISTIC, header=statistic_header, **technical_parameters): pass
class StochasticFile(File, variable=Variables.Technicals.STOCHASTIC, header=stochastic_header, **technical_parameters): pass
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


class TechnicalCalculator(Processor, formatter=technical_formatter):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        calculations = {variables["technical"]: calculation for variables, calculation in ODict(list(TechnicalCalculation)).items()}
        self.__calculations = {technical: calculation(*args, **kwargs) for technical, calculation in calculations.items()}

    def processor(self, contents, *args, **kwargs):
        bars = contents[Variables.Technicals.BARS]
        assert isinstance(bars, pd.DataFrame)
        technicals = ODict(list(self.calculate(bars, *args, **kwargs)))
        yield contents | technicals

    def calculate(self, bars, *args, **kwargs):
        for technical, calculation in self.calculations.items():
            dataframe = calculation(bars, *args, **kwargs)
            yield technical, dataframe

    @property
    def calculations(self): return self.__calculations



