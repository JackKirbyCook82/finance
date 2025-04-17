# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Valuation Objects
@author: Jack Kirby Cook

"""

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
class ValuationEquation(Equation, ABC, datatype=xr.DataArray, vectorize=True):
    τ = Variable.Dependent("τ", "tau", np.int32, function=lambda tτ, *, to: (tτ - to).days)

    wio = Variable.Independent("wio", "invest", np.float32, locator="invest")
    wbo = Variable.Independent("wbo", "borrow", np.float32, locator="borrow")
    wro = Variable.Independent("wro", "revenue", np.float32, locator="revenue")
    weo = Variable.Independent("weo", "expense", np.float32, locator="expense")
    wo = Variable.Independent("wo", "spot", np.float32, locator="spot")

    xo = Variable.Independent("xo", "underlying", np.float32, locator="underlying")
    qo = Variable.Independent("qo", "size", np.int32, locator="size")
    to = Variable.Constant("to", "date", Date, locator="date")
    tτ = Variable.Independent("tτ", "expire", Date, locator="expire")
    ρ = Variable.Constant("ρ", "discount", np.float32, locator="discount")

class ArbitrageEquation(ValuationEquation, ABC):
    npv = Variable.Dependent("npv", "npv", np.float32, function=lambda wτ, wo, τ, *, ρ: np.divide(wτ, np.power(ρ + 1, np.divide(τ, 365))) + wo)

class MinimumArbitrageEquation(ArbitrageEquation):
    wτ = Variable.Independent("wτ", "future", np.float32, locator="minimum")

class MaximumArbitrageEquation(ArbitrageEquation):
    wτ = Variable.Independent("wτ", "future", np.float32, locator="maximum")

# class ExpectedArbitrageEquation(ArbitrageEquation):
#     wτ = Variable.Independent("vτ", "future", np.float32, locator="expected")


class ValuationCalculation(Calculation, ABC, metaclass=RegistryMeta): pass
class ArbitrageCalculation(ValuationCalculation, ABC):
    def execute(self, strategies, *args, discount, date, **kwargs):
        with self.equation(strategies, discount=discount, date=date) as equation:
            yield equation.τ()
            yield equation.xo()
            yield equation.qo()
            yield equation.wo()
            yield equation.wτ()
            yield equation.wio()
            yield equation.wbo()
            yield equation.wro()
            yield equation.weo()
            yield equation.npv()

class MinimumArbitrageCalculation(ArbitrageCalculation, equation=MinimumArbitrageEquation, register=ValuationLocator(Variables.Valuations.Valuation.ARBITRAGE, Variables.Valuations.Scenario.MINIMUM)): pass
class MaximumArbitrageCalculation(ArbitrageCalculation, equation=MaximumArbitrageEquation, register=ValuationLocator(Variables.Valuations.Valuation.ARBITRAGE, Variables.Valuations.Scenario.MAXIMUM)): pass

# class ExpectedArbitrageCalculation(ArbitrageCalculation, equation=ExpectedArbitrageEquation, register=ValuationLocator(Variables.Valuations.Valuation.ARBITRAGE, Variables.Valuations.Scenario.EXPECTED)):
#     pass


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

class ArbitrageStacking(ValuationStacking, header=["npv", "future"], register=Variables.Valuations.Valuation.ARBITRAGE):
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



