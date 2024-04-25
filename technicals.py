# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 2024
@name:   Technical Objects
@author: Jack Kirby Cook

"""

import numpy as np
from abc import ABC
from scipy.signal import convolve
from collections import OrderedDict as ODict

from support.calculations import Calculation, equation, source, constant
from support.processes import Calculator
from support.pipelines import Processor
from support.files import Files

from finance.variables import Technicals

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["BarFile", "StatisticFile"]
__copyright__ = "Copyright 2024, Jack Kirby Cook"
__license__ = "MIT License"


history_index = {"ticker": str, "date": np.datetime64}
bars_columns = {"high": np.float32, "low": np.float32, "open": np.float32, "close": np.float32, "price": np.float32, "price": np.float32, "volume": np.float32}
stats_columns = {}


class BarFile(Files.Dataframe, variable="bars", index=history_index, columns=bars_columns): pass
class StatisticFile(Files.Dataframe, variable="statistics", index=history_index, columns=stats_columns): pass


class TechnicalCalculation(Calculation, ABC, fields=["technical"]):
    bar = source("bar", "stock", position=0, variables={"ti": "date", "xi": "price"})
    Δi = constant("Δi", "period", position="period")


class StatisticCalculation(TechnicalCalculation, technical=Technicals.STATISTIC):
    ri = equation("ri", "return", np.float32, domain=("bar.xi",), function=lambda xi: xi.rolling(window=2).apply(np.diff))
    μi = equation("μi", "trend", np.float32, domain=("ri", "Δi"), function=lambda ri, Δi: ri.rolling(window=Δi).apply(np.mean))
    δi = equation("δi", "volatility", np.float32, domain=("ri", "Δi"), function=lambda ri, Δi: ri.rolling(window=Δi).apply(np.std))

    def execute(self, feed, *args, period, **kwargs):
        yield self.μi(feed, period=period)
        yield self.δi(feed, period=period)


class StatisticCalculator(Calculator, Processor, calculations=ODict(list(StatisticCalculation)), title="Calculated"):
    def execute(self, contents, *args, **kwargs):
        pass



