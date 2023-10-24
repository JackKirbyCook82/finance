# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Valuation Objects
@author: Jack Kirby Cook

"""

import xarray as xr
from enum import IntEnum
from collections import namedtuple as ntuple

from support.pipelines import Calculator
from support.calculations import Calculation

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Valuation", "Valuations", "Calculations", "ValuationCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


Methods = IntEnum("Method", ["ARBITRAGE"], start=1)
class Valuation(ntuple("Valuation", "method")):
    def __str__(self): return "|".join([str(value.name).lower() for value in self if bool(value)])

Arbitrage = Valuation(Methods.ARBITRAGE)


class ValuationCalculation(Calculation): pass
class ArbitrageCalculation(Calculation): pass


Valuations = {"Arbitrage": Arbitrage}
Calculations = {"Arbitrage": ArbitrageCalculation}


class ValuationCalculator(Calculator):
    def execute(self, contents, *args, **kwargs):
        ticker, expire, datasets = contents
        assert isinstance(datasets, dict)
        assert all([isinstance(security, xr.Dataset) for security in datasets.values()])




