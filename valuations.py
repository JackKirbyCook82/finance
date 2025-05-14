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

    Δo = Variable.Independent("Δo", "delta", np.float32, locator="delta")
    Γo = Variable.Independent("Γo", "gamma", np.float32, locator="gamma")
    Θo = Variable.Independent("Θo", "theta", np.float32, locator="theta")
    Vo = Variable.Independent("Vo", "vega", np.float32, locator="vega")
    Po = Variable.Independent("Po", "rho", np.float32, locator="rho")

    xo = Variable.Independent("xo", "underlying", np.float32, locator="underlying")
    qo = Variable.Independent("qo", "size", np.int32, locator="size")
    to = Variable.Constant("to", "date", Date, locator="date")
    tτ = Variable.Independent("tτ", "expire", Date, locator="expire")
    ρ = Variable.Constant("ρ", "discount", np.float32, locator="discount")

class ArbitrageEquation(ValuationEquation, ABC):
    vlo = Variable.Dependent("vlo", ("npv", Variables.Valuations.Scenario.MINIMUM), np.float32, function=lambda wlτ, wo, τ, *, ρ: np.divide(wlτ, np.power(ρ + 1, np.divide(τ, 365))) + wo)
    veo = Variable.Dependent("veo", ("npv", Variables.Valuations.Scenario.EXPECTED), np.float32, function=lambda weτ, wo, τ, *, ρ: np.divide(weτ, np.power(ρ + 1, np.divide(τ, 365))) + wo)
    vho = Variable.Dependent("vho", ("npv", Variables.Valuations.Scenario.MAXIMUM), np.float32, function=lambda whτ, wo, τ, *, ρ: np.divide(whτ, np.power(ρ + 1, np.divide(τ, 365))) + wo)
    wlτ = Variable.Independent("wlτ", ("future", Variables.Valuations.Scenario.MINIMUM), np.float32, locator="minimum")
    weτ = Variable.Independent("weτ", ("future", Variables.Valuations.Scenario.EXPECTED), np.float32, locator="expected")
    whτ = Variable.Independent("whτ", ("future", Variables.Valuations.Scenario.MAXIMUM), np.float32, locator="maximum")


class ValuationCalculation(Calculation, ABC, metaclass=RegistryMeta): pass
class ArbitrageCalculation(ValuationCalculation, ABC, equation=ArbitrageEquation, register=Variables.Valuations.Valuation.ARBITRAGE):
    def execute(self, strategies, *args, discount, date, **kwargs):
        with self.equation(strategies, discount=discount, date=date) as equation:
            yield equation.vlo()
            yield equation.veo()
            yield equation.vho()
            yield equation.wlτ()
            yield equation.weτ()
            yield equation.whτ()
            yield equation.wro()
            yield equation.weo()
            yield equation.wio()
            yield equation.wbo()
            yield equation.wo()
            yield equation.xo()
            yield equation.qo()
            yield equation.τ()
#            yield equation.Δo()
#            yield equation.Γo()
#            yield equation.Θo()
#            yield equation.Vo()
#            yield equation.Po()

class ValuationCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, valuation, **kwargs):
        assert valuation in Variables.Valuations.Valuation
        super().__init__(*args, **kwargs)
        self.__calculation = ValuationCalculation[valuation](*args, **kwargs)
        self.__valuation = valuation

    def execute(self, strategies, *args, **kwargs):
        assert isinstance(strategies, xr.Dataset)
        if self.empty(strategies, "size"): return
        valuations = self.calculate(strategies, *args, **kwargs)
        settlements = self.groups(valuations, by=Querys.Settlement)
        settlements = ",".join(list(map(str, settlements)))
        size = self.size(valuations)
        self.console(f"{str(settlements)}|{str(self.valuation)}[{int(size):.0f}]")
        if self.empty(valuations): return
        yield valuations

    def calculate(self, strategies, *args, **kwargs):
        valuations = self.calculation(strategies, *args, **kwargs)
        valuations = valuations.to_dataframe().dropna(how="all", inplace=False)
        valuations = valuations.reset_index(drop=False, inplace=False)
        options = [option for option in list(map(str, Securities.Options)) if option not in valuations.columns]
        for option in options: valuations[option] = np.NaN
        valuations["breakeven"] = valuations["spot"] - valuations[("npv", Variables.Valuations.Scenario.MINIMUM)]
        valuations["valuation"] = self.valuation
        columns = {column: (column, "") for column in valuations.columns if not isinstance(column, tuple)}
        valuations = valuations.rename(columns=columns, inplace=False)
        valuations.columns = pd.MultiIndex.from_tuples(valuations.columns)
        valuations.columns.names = ["axis", "scenario"]
        return valuations

    @property
    def calculation(self): return self.__calculation
    @property
    def valuation(self): return self.__valuation



