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
from datetime import date as Date
from collections import namedtuple as ntuple

from finance.variables import Variables, Querys, Securities
from support.calculations import Calculation, Equation, Variable
from support.mixins import Emptying, Sizing, Partition, Logging
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ValuationCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class ValuationLocator(ntuple("Locator", "valuation scenario")): pass
class ValuationEquation(Equation, ABC):
    rev = Variable("rev", "revenue", np.float32, xr.DataArray, vectorize=True, function=lambda vo, vτ: np.abs(+ np.maximum(vo, 0) + np.maximum(vτ, 0)))
    exp = Variable("exp", "expense", np.float32, xr.DataArray, vectorize=True, function=lambda vo, vτ: np.abs(- np.minimum(vo, 0) - np.minimum(vτ, 0)))
    tau = Variable("tau", "days", np.int32, xr.DataArray, vectorize=True, function=lambda to, tτ: (tτ - to).days)

    vo = Variable("vo", "spot", np.float32, xr.DataArray, locator="spot")
    tτ = Variable("tτ", "expire", Date, xr.DataArray, locator="expire")
    to = Variable("to", "date", Date, types.NoneType, locator="date")
    ρ = Variable("ρ", "discount", np.float32, types.NoneType, locator="discount")

class ArbitrageEquation(ValuationEquation, ABC):
    npv = Variable("npv", "npv", np.float32, xr.DataArray, vectorize=True, function=lambda vτ, vo, tau, ρ: np.divide(vτ, np.power(ρ + 1, np.divide(tau, 365))) + vo)
    roi = Variable("roi", "roi", np.float32, xr.DataArray, vectorize=True, function=lambda exp, rev, tau: np.divide(rev - exp, tau) / exp)
    apy = Variable("apy", "apy", np.float32, xr.DataArray, vectorize=True, function=lambda roi, tau: np.power((roi + 1), 365))

class MinimumArbitrageEquation(ArbitrageEquation):
    vτ = Variable("vτ", "future", np.float32, xr.DataArray, locator="minimum")

class MaximumArbitrageEquation(ArbitrageEquation):
    vτ = Variable("vτ", "future", np.float32, xr.DataArray, locator="maximum")


class ValuationCalculation(Calculation, ABC, metaclass=RegistryMeta): pass
class ArbitrageCalculation(ValuationCalculation, ABC):
    def execute(self, strategies, *args, discount, date, **kwargs):
        with self.equation(strategies, discount=discount, date=date) as equation:
            yield equation.vo()
            yield equation.vτ()
            yield equation.npv()
            yield equation.apy()

class MinimumArbitrageCalculation(ArbitrageCalculation, equation=MinimumArbitrageEquation, register=ValuationLocator(Variables.Valuations.Valuation.ARBITRAGE, Variables.Valuations.Scenario.MINIMUM)): pass
class MaximumArbitrageCalculation(ArbitrageCalculation, equation=MaximumArbitrageEquation, register=ValuationLocator(Variables.Valuations.Valuation.ARBITRAGE, Variables.Valuations.Scenario.MAXIMUM)): pass


class ValuationStacking(ABC, metaclass=RegistryMeta):
    def __init_subclass__(cls, *args, header=[], **kwargs): cls.__header__ = header
    def __init__(self, *args, **kwargs): pass
    def __call__(self, valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        index = set(valuations.columns) - ({"scenario"} | set(self.header))
        valuations = valuations.pivot(index=index, columns=["scenario"])
        valuations = valuations.reset_index(drop=False, inplace=False)
        return valuations

    @property
    def header(self): return type(self).__header__

class ArbitrageStacking(ValuationStacking, header=["apy", "npv", "future"], register=Variables.Valuations.Valuation.ARBITRAGE):
    pass


class ValuationCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, valuation, **kwargs):
        assert valuation in Variables.Valuations.Valuation
        super().__init__(*args, **kwargs)
        calculations = {locator.scenario: calculation for locator, calculation in dict(ValuationCalculation).items() if locator.valuation == valuation}
        self.__calculations = {scenario: calculation(*args, **kwargs) for scenario, calculation in calculations.items()}
        self.__stacking = dict(ValuationStacking)[valuation](*args, **kwargs)
        self.__valuation = valuation

    def execute(self, strategies, *args, **kwargs):
        assert isinstance(strategies, xr.Dataset)
        if self.empty(strategies, "size"): return
        valuations = self.calculate(strategies, *args, **kwargs)
        valuations = self.stacking(valuations, *args, **kwargs)
        settlements = self.groups(valuations, by=Querys.Settlement)
        settlements = ",".join(list(map(str, settlements)))
        size = self.size(valuations)
        self.console(f"{str(settlements)}|{str(self.valuation)}[{int(size):.0f}]")
        if self.empty(valuations): return
        yield valuations

    def calculate(self, strategies, *args, **kwargs):
        valuations = dict(self.calculator(strategies, *args, **kwargs))
        valuations = pd.concat(list(valuations.values()), axis=0)
        columns = [option for option in list(map(str, Securities.Options)) if option not in valuations.columns]
        for column in columns: valuations[column] = np.NaN
        return valuations

    def calculator(self, strategies, *args, **kwargs):
        for scenario, calculation in self.calculations.items():
            valuations = calculation(strategies, *args, **kwargs)
            assert isinstance(valuations, xr.Dataset)
            valuations = valuations.assign_coords({"valuation": xr.Variable("valuation", [self.valuation]).squeeze("valuation")})
            valuations = valuations.assign_coords({"scenario": xr.Variable("scenario", [scenario]).squeeze("scenario")}).expand_dims("scenario")
            valuations = valuations.drop_vars(list(Variables.Securities.Security), errors="ignore")
            valuations = valuations.expand_dims(list(set(iter(valuations.coords)) - set(iter(valuations.dims))))
            valuations = valuations.to_dataframe().dropna(how="all", inplace=False)
            valuations = valuations.reset_index(drop=False, inplace=False)
            yield scenario, valuations

    @property
    def calculations(self): return self.__calculations
    @property
    def stacking(self): return self.__stacking
    @property
    def valuation(self): return self.__valuation



