# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Valuation Objects
@author: Jack Kirby Cook

"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from enum import IntEnum
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

from support.pipelines import Processor, Calculator, Saver
from support.calculations import Calculation, equation, source, constant
from support.dispatchers import typedispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Valuation", "Valuations", "Calculations", "ValuationProcessor", "ValuationCalculator", "ValuationSaver"]
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

class ValuationsMeta(type):
    def __iter__(cls): return iter([MinimumArbitrage, MaximumArbitrage])
    def __getitem__(cls, indexkey): return cls.retrieve(indexkey)

    @typedispatcher
    def retrieve(cls, indexkey): raise TypeError(type(indexkey).__name__)
    @retrieve.register(int)
    def integer(cls, index): return {int(valuation): valuation for valuation in iter(cls)}[index]
    @retrieve.register(str)
    def string(cls, string): return {int(valuation): valuation for valuation in iter(cls)}[str(string).lower()]

    class Arbitrage:
        Current = CurrentArbitrage
        Minimum = MinimumArbitrage
        Maximum = MaximumArbitrage

class Valuations(object, metaclass=ValuationsMeta):
    pass


class ValuationCalculation(Calculation):
    Λ = source("Λ", "valuation", position=0, variables={"τ": "tau", "q": "size", "i": "interest"})
    ρ = constant("ρ", "discount", position="discount")

    inc = equation("inc", "income", np.float32, domain=("Λ.vo", "Λ.vτ"), function=lambda vo, vτ: + np.maximum(vo, 0) + np.maximum(vτ, 0))
    exp = equation("exp", "cost", np.float32, domain=("Λ.vo", "Λ.vτ"), function=lambda vo, vτ: - np.minimum(vo, 0) - np.minimum(vτ, 0))
    apy = equation("apy", "apy", np.float32, domain=("r", "Λ.τ"), function=lambda r, τ: np.power(r + 1, np.power(τ / 365, -1)) - 1)
    npv = equation("npv", "npv", np.float32, domain=("π", "Λ.τ", "ρ"), function=lambda π, τ, ρ: π * np.power(ρ / 365 + 1, τ))
    π = equation("π", "profit", np.float32, domain=("inc", "exp"), function=lambda inc, exp: inc - exp)
    r = equation("r", "return", np.float32, domain=("π", "exp"), function=lambda π, exp: π / exp)

    def execute(self, dataset, *args, discount, **kwargs):
        yield self["Λ"].τ(dataset, discount=discount)
        yield self.inc(dataset, discount=discount)
        yield self.exp(dataset, discount=discount)
        yield self.npv(dataset, discount=discount)
        yield self.apy(dataset, discount=discount)

class ArbitrageCalculation(ValuationCalculation):
    Λ = source("Λ", "arbitrage", position=0, variables={"vo": "spot", "vτ": "future"})

class MinimumCalculation(ArbitrageCalculation):
    Λ = source("Λ", "minimum", position=0, variables={"vτ": "minimum"})

class MaximumCalculation(ArbitrageCalculation):
    Λ = source("Λ", "maximum", position=0, variables={"vτ": "maximum"})

class CalculationsMeta(type):
    def __iter__(cls):
        contents = {Valuations.Arbitrage.Minimum: MinimumCalculation, Valuations.Arbitrage.Maximum: MaximumCalculation}
        return ((key, value) for key, value in contents.items())

    class Arbitrage:
        Minimum = MinimumCalculation
        Maximum = MaximumCalculation

class Calculations(object, metaclass=CalculationsMeta):
    pass


class ValuationCalculator(Calculator, calculations=ODict(list(iter(Calculations)))):
    def execute(self, contents, *args, **kwargs):
        current, ticker, expire, strategy, dataset = contents
        assert isinstance(dataset, xr.Dataset)
        for valuation, calculation in self.calculations.items():
            results = calculation(dataset, *args, **kwargs)
            yield current, ticker, expire, strategy, valuation, results


class ValuationProcessor(Processor):
    def execute(self, contents, *args, **kwargs):
        current, ticker, expire, strategy, valuation, dataset = contents
        assert isinstance(dataset, xr.Dataset)
        dataframe = self.parser(dataset, *args, **kwargs)
        dataframe = self.filter(dataframe, *args, **kwargs)
        yield current, ticker, expire, strategy, valuation, dataframe

    @staticmethod
    def parser(dataset, *args, **kwargs):
        dataframe = dataset.to_dataframe()
        dataframe = dataframe.reset_index(drop=False, inplace=False)
        return dataframe

    @staticmethod
    def filter(dataframe, *args, apy=None, cost=None, size=None, interest=None, **kwargs):
        dataframe = dataframe.where(dataframe["apy"] >= apy) if bool(apy) else dataframe
        dataframe = dataframe.where(dataframe["cost"] >= cost) if bool(cost) else dataframe
        dataframe = dataframe.where(dataframe["size"] >= size) if bool(size) else dataframe
        dataframe = dataframe.where(dataframe["interest"] >= interest) if bool(interest) else dataframe
        return dataframe


class ValuationSaver(Saver):
    def execute(self, contents, *args, **kwargs):
        current, ticker, expire, strategy, valuation, dataframe = contents
        assert isinstance(dataframe, pd.DataFrame)
        current_folder = os.path.join(self.repository, str(current.strftime("%Y%m%d_%H%M%S")))
        if not os.path.isdir(current_folder):
            os.mkdir(current_folder)
        valuation_folder = os.path.join(current_folder, str(valuation).replace("|", "_"))
        if not os.path.isdir(valuation_folder):
            os.mkdir(valuation_folder)
        strategy_folder = os.path.join(valuation_folder, str(strategy).replace("|", "_"))
        if not os.path.isdir(strategy_folder):
            os.mkdir(strategy_folder)
        filename = "_".join([str(ticker), str(expire.strftime("%Y%m%d"))]) + ".csv"
        file = os.path.join(strategy_folder, filename)
        self.write(dataframe, file=file, mode="a")



