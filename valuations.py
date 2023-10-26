# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Valuation Objects
@author: Jack Kirby Cook

"""

import numpy as np
import xarray as xr

from support.pipelines import Calculator
from support.calculations import Calculation, equation

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = []
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


class ValuationCalculation(Calculation, variables={"τ": "tau", "w": "price", "k": "strike", "x": "time", "q": "size", "i": "interest"}, sources={"ρ": "discount"}):
    inc = equation("income", np.float32, function=lambda vo, vτ: + np.maximum(vo, 0) + np.maximum(vτ, 0))
    exp = equation("expense", np.float32, function=lambda vo, vτ: - np.minimum(vo, 0) - np.minimum(vτ, 0))
    apy = equation("apy", np.float32, function=lambda r, τ: np.power(r + 1, np.power(τ / 365, -1)) - 1)
    npv = equation("npv", np.float32, function=lambda π, τ, ρ: π * np.power(ρ / 365 + 1, τ))
    π = equation("profit", np.float32, function=lambda inc, exp: inc - exp)
    r = equation("return", np.float32, function=lambda π, exp: π / exp)

class ArbitrageCalculation(ValuationCalculation, variables={"vo": "spot", "vτ": "future"}): pass
class CurrentCalculation(ArbitrageCalculation, variables={"vτ": "current"}): pass
class MinimumCalculation(ArbitrageCalculation, variables={"vτ": "minimum"}): pass
class MaximumCalculation(ArbitrageCalculation, variables={"vτ": "maximum"}): pass


calculations = {}
class ValuationCalculator(Calculator, calculations=calculations):
    def execute(self, contents, *args, **kwargs):
        ticker, expire, datasets = contents
        assert isinstance(datasets, dict)
        assert all([isinstance(security, xr.Dataset) for security in datasets.values()])

        ###


