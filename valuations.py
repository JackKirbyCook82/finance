# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Valuation Objects
@author: Jack Kirby Cook

"""

import types
import numpy as np
import pandas as pd
import xarray as xr
from abc import ABC
from collections import namedtuple as ntuple

from finance.variables import Variables, Querys
from support.mixins import Emptying, Sizing, Partition
from support.calculations import Calculation, Equation, Variable
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ValuationCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class ValuationLocator(ntuple("Locator", "valuation scenario")): pass
class ValuationEquation(Equation, ABC):
    tau = Variable("tau", "tau", np.int32, xr.DataArray, vectorize=True, function=lambda to, tτ: np.timedelta64(np.datetime64(tτ, "ns") - np.datetime64(to, "ns"), "D") / np.timedelta64(1, "D"))
    inc = Variable("inc", "income", np.float32, xr.DataArray, vectorize=True, function=lambda vo, vτ: + np.maximum(vo, 0) + np.maximum(vτ, 0))
    exp = Variable("exp", "cost", np.float32, xr.DataArray, vectorize=True, function=lambda vo, vτ: - np.minimum(vo, 0) - np.minimum(vτ, 0))

    xo = Variable("xo", "underlying", np.float32, xr.DataArray, locator="underlying")
    tτ = Variable("tτ", "expire", np.datetime64, xr.DataArray, locator="expire")
    to = Variable("to", "current", np.datetime64, xr.DataArray, locator="current")
    vo = Variable("vo", "spot", np.float32, xr.DataArray, locator="spot")
    ρ = Variable("ρ", "discount", np.float32, types.NoneType, locator="discount")

class ArbitrageEquation(ValuationEquation, ABC):
    irr = Variable("irr", "irr", np.float32, xr.DataArray, vectorize=True, function=lambda inc, exp, tau: np.power(np.divide(inc, exp), np.power(tau, -1)) - 1)
    npv = Variable("npv", "npv", np.float32, xr.DataArray, vectorize=True, function=lambda inc, exp, tau, ρ: np.divide(inc, np.power(1 + ρ, tau / 365)) - exp)
    apy = Variable("apy", "apy", np.float32, xr.DataArray, vectorize=True, function=lambda irr, tau: np.power(irr + 1, np.power(tau / 365, -1)) - 1)

class MinimumArbitrageEquation(ArbitrageEquation):
    vτ = Variable("vτ", "minimum", np.float32, xr.DataArray, locator="minimum")

class MaximumArbitrageEquation(ArbitrageEquation):
    vτ = Variable("vτ", "maximum", np.float32, xr.DataArray, locator="maximum")


class ValuationCalculation(Calculation, ABC, metaclass=RegistryMeta): pass
class ArbitrageCalculation(ValuationCalculation, ABC):
    def execute(self, strategies, *args, discount, **kwargs):
        with self.equation(strategies, discount=discount) as equation:
            yield strategies["size"]
            yield strategies["underlying"]
            yield strategies["current"]
            yield equation.exp()
            yield equation.npv()
            yield equation.apy()

class MinimumArbitrageCalculation(ArbitrageCalculation, equation=MinimumArbitrageEquation, register=ValuationLocator(Variables.Valuations.Valuation.ARBITRAGE, Variables.Valuations.Scenario.MINIMUM)): pass
class MaximumArbitrageCalculation(ArbitrageCalculation, equation=MaximumArbitrageEquation, register=ValuationLocator(Variables.Valuations.Valuation.ARBITRAGE, Variables.Valuations.Scenario.MAXIMUM)): pass


class ValuationCalculator(Sizing, Emptying, Partition, query=Querys.Settlement, title="Calculated"):
    def __init__(self, *args, valuation, **kwargs):
        assert valuation in Variables.Valuations.Valuation
        super().__init__(*args, **kwargs)
        calculations = {locator.scenario: calculation for locator, calculation in dict(ValuationCalculation).items() if locator.valuation == valuation}
        self.__calculations = {scenario: calculation(*args, **kwargs) for scenario, calculation in calculations.items()}
        self.__valuation = valuation

    def execute(self, strategies, *args, **kwargs):
        assert isinstance(strategies, (list, xr.Dataset))
        if self.empty(strategies, "size"): return
        for settlement, dataset in self.partition(strategies):
            valuations = self.calculate(dataset, *args, **kwargs)
            size = self.size(valuations)
            string = f"{str(settlement)}|{str(self.valuation)}[{int(size):.0f}]"
            self.console(string)
            if self.empty(valuations): continue
            yield valuations

    def calculate(self, strategies, *args, **kwargs):
        valuations = dict(self.calculator(strategies, *args, **kwargs))
        valuations = pd.concat(list(valuations.values()), axis=0)
        return valuations

    def calculator(self, strategies, *args, **kwargs):
        assert isinstance(strategies, xr.Dataset)
        for scenario, calculation in self.calculations.items():
            valuations = calculation(strategies, *args, **kwargs)
            assert isinstance(valuations, xr.Dataset)
            valuations = valuations.assign_coords({"valuations": xr.Variable("valuations", [self.valuation]).squeeze("valuations")})
            valuations = valuations.assign_coords({"scenario": xr.Variable("scenario", [scenario]).squeeze("scenario")}).expand_dims("scenario")
            valuations = valuations.drop_vars(list(Variables.Securities.Security), errors="ignore")
            valuations = valuations.expand_dims(list(set(iter(valuations.coords)) - set(iter(valuations.dims))))
            valuations = valuations.to_dataframe().dropna(how="all", inplace=False)
            valuations = valuations.reset_index(drop=False, inplace=False)
            yield scenario, valuations

    @property
    def calculations(self): return self.__calculations
    @property
    def valuation(self): return self.__valuation



