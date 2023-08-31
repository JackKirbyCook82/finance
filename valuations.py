# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Valuation Objects
@author: Jack Kirby Cook

"""

import numpy as np
import xarray as xr

from support.pipelines import Calculator
from support.calculations import Calculation, feed, equation

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ValuationCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


class ValuationCalculation(Calculation):
    ρ = feed("discount", np.float16, variable="discount")
    to = feed("date", np.datetime64, variable="date")
    tτ = feed("expire", np.datetime64, variable="expire")
    vo = feed("spot", np.float16, variable="spot")
    vτ = feed("value", np.float16, variable="value")

    τau = equation("τau", np.int16, function=lambda tτ, to: np.timedelta64(np.datetime64(tτ, "ns") - np.datetime64(to, "ns"), "D") / np.timedelta64(1, "D"))
    inc = equation("income", np.float32, function=lambda vo, vτ: + np.maximum(vo, 0) + np.maximum(vτ, 0))
    cost = equation("cost", np.float32, function=lambda vo, vτ: - np.minimum(vo, 0) - np.minimum(vτ, 0))
    apy = equation("apy", np.float32, function=lambda r, τau: np.power(r + 1, np.power(τau / 365, -1)) - 1)
    npv = equation("npv", np.float32, function=lambda π, τau, ρ: π * np.power(ρ / 365 + 1, τau))
    π = equation("profit", np.float32, function=lambda inc, cost: inc - cost)
    r = equation("return", np.float32, function=lambda π, cost: π / cost)

    def __call__(self, strategies, *args, discount, fees, **kwargs):
        assert isinstance(strategies, xr.Dataset)
        strategies["tau"] = self.τau(strategies)
        strategies["cost"] = self.cost(strategies)
        strategies["apy"] = self.apy(strategies)
        return strategies


class ValuationCalculator(Calculator, calculations=[ValuationCalculation]):
    def execute(self, contents, *args, **kwargs):
        ticker, expire, strategy, strategies = contents
        assert isinstance(strategies, xr.Dataset)
        for calculation in iter(self.calculations):
            valuations = calculation(strategies, *args, **kwargs)
            yield ticker, expire, strategy, valuations


