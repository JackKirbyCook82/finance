# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Valuation Objects
@author: Jack Kirby Cook

"""

import numpy as np
import xarray as xr
from enum import IntEnum
from collections import namedtuple as ntuple

from support.pipelines import Calculator
from support.calculations import Calculation, equation

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Valuation", "Valuations", "Calculations", "ValuationCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


Basis = IntEnum("Basis", ["ARBITRAGE"], start=1)
Scenario = IntEnum("Scenario", ["CURRENT", "MINIMUM", "MAXIMUM"], start=1)
class Valuation(ntuple("Valuation", "basis scenario")):
    def __str__(self): return "|".join([str(value.name).lower() for value in self if bool(value)])

CurrentArbitrage = Valuation(Basis.ARBITRAGE, Scenario.CURRENT)
MinimumArbitrage = Valuation(Basis.ARBITRAGE, Scenario.MINIMUM)
MaximumArbitrage = Valuation(Basis.ARBITRAGE, Scenario.MAXIMUM)

class Valuations:
    class Arbitrage:
        Current = CurrentArbitrage
        Minimum = MinimumArbitrage
        Maximum = MaximumArbitrage


class ValuationCalculation(Calculation, vars={"τ": "tau", "w": "price", "k": "strike", "x": "time", "q": "size", "i": "interest"}, parms={"ρ": "discount"}):
    inc = equation("inc", "income", np.float32, domain=("0.vo", "0.vτ"), function=lambda vo, vτ: + np.maximum(vo, 0) + np.maximum(vτ, 0))
    exp = equation("exp", "expense", np.float32, domain=("0.vo", "0.vτ"), function=lambda vo, vτ: - np.minimum(vo, 0) - np.minimum(vτ, 0))
    apy = equation("apy", "yield", np.float32, domain=("0.r", "0.τ"), function=lambda r, τ: np.power(r + 1, np.power(τ / 365, -1)) - 1)
    npv = equation("npv", "value", np.float32, domain=("0.π", "0.τ"), function=lambda π, τ, ρ: π * np.power(ρ / 365 + 1, τ))
    π = equation("π", "profit", np.float32, domain=("0.inc", "0.exp"), function=lambda inc, exp: inc - exp)
    r = equation("r", "return", np.float32, domain=("0.π", "0.exp"), function=lambda π, exp: π / exp)

class ArbitrageCalculation(ValuationCalculation, vars={"vo": "spot", "vτ": "future"}): pass
class CurrentCalculation(ArbitrageCalculation, vars={"vτ": "current"}): pass
class MinimumCalculation(ArbitrageCalculation, vars={"vτ": "minimum"}): pass
class MaximumCalculation(ArbitrageCalculation, vars={"vτ": "maximum"}): pass

class Calculations:
    class Arbitrage:
        Current = CurrentCalculation
        Minimum = MinimumCalculation
        Maximum = MaximumCalculation


calculations = {Valuations.Arbitrage.Minimum: Calculations.Arbitrage.Minimum, Valuations.Arbitrage.Maximum: Calculations.Arbitrage.Maximum}
calculations.update({Valuations.Arbitrage.Current: Calculations.Arbitrage.Current})
class ValuationCalculator(Calculator, calculations=calculations):
    def execute(self, contents, *args, **kwargs):
        ticker, expire, strategy, datasets = contents
        assert isinstance(datasets, dict)
        assert all([isinstance(security, xr.Dataset) for security in datasets.values()])
        for valuation, calculation in self.calculations.items():
            valuations = calculation(*args, **datasets, **kwargs)
            yield ticker, expire, strategy, valuation, valuations



