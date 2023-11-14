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

from support.pipelines import Calculator, Saver, Loader
from support.calculations import Calculation, equation, source

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Valuation", "Valuations", "Calculations", "ValuationCalculator", "ValuationSaver", "ValuationLoader"]
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


class ValuationCalculation(Calculation):
    v = source("v", "valuation", position=0, variables={"τ": "tau", "w": "price", "k": "strike", "s": "time", "q": "size", "i": "interest"})
#    ρ = source("ρ", "discount", position="discount", variables={})

    inc = equation("inc", "income", np.float32, domain=("v.vo", "v.vτ"), function=lambda vo, vτ: + np.maximum(vo, 0) + np.maximum(vτ, 0))
    exp = equation("exp", "cost", np.float32, domain=("v.vo", "v.vτ"), function=lambda vo, vτ: - np.minimum(vo, 0) - np.minimum(vτ, 0))
    apy = equation("apy", "yield", np.float32, domain=("v.r", "v.τ"), function=lambda r, τ: np.power(r + 1, np.power(τ / 365, -1)) - 1)
    npv = equation("npv", "value", np.float32, domain=("v.π", "v.τ"), function=lambda π, τ, ρ: π * np.power(ρ / 365 + 1, τ))
    π = equation("π", "profit", np.float32, domain=("v.inc", "v.exp"), function=lambda inc, exp: inc - exp)
    r = equation("r", "return", np.float32, domain=("v.π", "v.exp"), function=lambda π, exp: π / exp)

    def execute(self, dataset, *args, **kwargs):
        dataset["tau"] = self.τ(*args, **kwargs)
        dataset["cost"] = self.exp(*args, **kwargs)
        dataset["apy"] = self.apy(*args, **kwargs)
        dataset["npv"] = self.npv(*args, **kwargs)

class ArbitrageCalculation(ValuationCalculation):
    v = source("v", "arbitrage", position=0, variables={"vo": "spot", "vτ": "future"})

class CurrentCalculation(ArbitrageCalculation):
    v = source("v", "current", position=0, variables={"vτ": "current"})

class MinimumCalculation(ArbitrageCalculation):
    v = source("v", "minimum", position=0, variables={"vτ": "minimum"})

class MaximumCalculation(ArbitrageCalculation):
    v = source("v", "maximum", position=0, variables={"vτ": "maximum"})

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
            results = calculation(*args, **datasets, **kwargs)
            yield ticker, expire, strategy, valuation, results


class ValuationSaver(Saver):
    def execute(self, contents, *args, **kwargs):
        ticker, expire, strategy, valuation, datasets = contents
        assert isinstance(datasets, xr.Dataset)


class ValuationLoader(Loader):
    def execute(self, *args, **kwargs):
        pass


