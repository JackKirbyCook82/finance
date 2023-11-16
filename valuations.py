# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Valuation Objects
@author: Jack Kirby Cook

"""

import os
import numpy as np
import xarray as xr
from enum import IntEnum
from collections import namedtuple as ntuple

from support.pipelines import Calculator, Saver
from support.calculations import Calculation, equation, source, constant

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Valuation", "Valuations", "Calculations", "ValuationCalculator", "ValuationSaver"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


Basis = IntEnum("Basis", ["ARBITRAGE"], start=1)
Scenario = IntEnum("Scenario", ["CURRENT", "MINIMUM", "MAXIMUM"], start=1)
class Valuation(ntuple("Valuation", "basis scenario")):
    def __str__(self): return "|".join([str(value.name).lower() for value in self if bool(value)])
    def __int__(self): return int(self.basis) * 10 + int(self.scenario) * 1

CurrentArbitrage = Valuation(Basis.ARBITRAGE, Scenario.CURRENT)
MinimumArbitrage = Valuation(Basis.ARBITRAGE, Scenario.MINIMUM)
MaximumArbitrage = Valuation(Basis.ARBITRAGE, Scenario.MAXIMUM)

class Valuations:
    class Arbitrage:
        Current = CurrentArbitrage
        Minimum = MinimumArbitrage
        Maximum = MaximumArbitrage

    @classmethod
    def fromInt(cls, integer): pass
    @classmethod
    def fromStr(cls, string): pass


class ValuationCalculation(Calculation):
    Λ = source("Λ", "valuation", position=0, variables={"τ": "tau", "w": "price", "k": "strike", "s": "time", "q": "size", "i": "interest"})
    ρ = constant("ρ", "discount", position="discount")

    inc = equation("inc", "income", np.float32, domain=("v.vo", "v.vτ"), function=lambda vo, vτ: + np.maximum(vo, 0) + np.maximum(vτ, 0))
    exp = equation("exp", "cost", np.float32, domain=("v.vo", "v.vτ"), function=lambda vo, vτ: - np.minimum(vo, 0) - np.minimum(vτ, 0))
    apy = equation("apy", "yield", np.float32, domain=("v.r", "v.τ"), function=lambda r, τ: np.power(r + 1, np.power(τ / 365, -1)) - 1)
    npv = equation("npv", "value", np.float32, domain=("v.π", "v.τ"), function=lambda π, τ, ρ: π * np.power(ρ / 365 + 1, τ))
    π = equation("π", "profit", np.float32, domain=("v.inc", "v.exp"), function=lambda inc, exp: inc - exp)
    r = equation("r", "return", np.float32, domain=("v.π", "v.exp"), function=lambda π, exp: π / exp)

    def execute(self, dataset, *args, **kwargs):
        yield self.τ(*args, **kwargs)
        yield self.exp(*args, **kwargs)
        yield self.apy(*args, **kwargs)
        yield self.npv(*args, **kwargs)

class ArbitrageCalculation(ValuationCalculation):
    Λ = source("Λ", "arbitrage", position=0, variables={"vo": "spot", "vτ": "future"})

class CurrentCalculation(ArbitrageCalculation):
    Λ = source("Λ", "current", position=0, variables={"vτ": "current"})

class MinimumCalculation(ArbitrageCalculation):
    Λ = source("Λ", "minimum", position=0, variables={"vτ": "minimum"})

class MaximumCalculation(ArbitrageCalculation):
    Λ = source("Λ", "maximum", position=0, variables={"vτ": "maximum"})

class Calculations:
    class Arbitrage:
        Current = CurrentCalculation
        Minimum = MinimumCalculation
        Maximum = MaximumCalculation


calculations = {Valuations.Arbitrage.Minimum: Calculations.Arbitrage.Minimum, Valuations.Arbitrage.Maximum: Calculations.Arbitrage.Maximum}
calculations.update({Valuations.Arbitrage.Current: Calculations.Arbitrage.Current})
class ValuationCalculator(Calculator, calculations=calculations):
    def execute(self, contents, *args, **kwargs):
        current, ticker, expire, strategy, datasets = contents
        assert isinstance(datasets, dict)
        assert all([isinstance(security, xr.Dataset) for security in datasets.values()])
        for valuation, calculation in self.calculations.items():
            results = calculation(*args, **datasets, **kwargs)
            yield current, ticker, expire, strategy, valuation, results


class ValuationSaver(Saver):
    def execute(self, contents, *args, **kwargs):
        current, ticker, expire, strategy, valuation, dataset = contents
        assert isinstance(dataset, xr.Dataset)
        folder = os.path.join(self.repository, str(valuation).replace("|", "_"))
        if not os.path.isdir(folder):
            os.mkdir(folder)
        folder = os.path.join(folder, str(strategy).replace("|", "_"))
        if not os.path.isdir(folder):
            os.mkdir(folder)
        filename = str(ticker) + str(expire.strftime("%Y%m%d")) + ".csv"
        file = os.path.join(folder, filename)
        dataframe = dataset.to_dask_dataframe()
        dataframe = dataframe.reset_index(drop=False, inplace=False)
        self.write(dataframe, file=file, mode="a")



