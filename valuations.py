# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Valuation Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
import xarray as xr
from datetime import date as Date

from finance.variables import Variables, Querys, Securities
from support.calculations import Calculation, Equation, Variable
from support.mixins import Emptying, Sizing, Partition, Logging
from support.decorators import TypeDispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ValuationCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class ValuationEquation(Equation, datatype=xr.DataArray, vectorize=True):
    τ = Variable.Dependent("τ", "tau", np.int32, function=lambda tτ, *, to: (tτ - to).days)

    xo = Variable.Independent("xo", "underlying", np.float32, locator="underlying")
    wo = Variable.Independent("wo", "spot", np.float32, locator="spot")
    qo = Variable.Independent("qo", "size", np.int32, locator="size")
    tτ = Variable.Independent("tτ", "expire", Date, locator="expire")
    to = Variable.Constant("to", "current", Date, locator="current")
    ρ = Variable.Constant("ρ", "discount", np.float32, locator="discount")

    def execute(self, *args, **kwargs):
        yield from super().execute(*args, **kwargs)
        yield self.xo()
        yield self.qo()
        yield self.τ()


class PayoffEquation(ValuationEquation):
    vlo = Variable.Dependent("vlo", ("npv", Variables.Scenario.MINIMUM), np.float32, function=lambda wlτ, wo, τ, *, ρ: np.divide(wlτ, np.power(ρ + 1, np.divide(τ, 365))) + wo)
    vho = Variable.Dependent("vho", ("npv", Variables.Scenario.MAXIMUM), np.float32, function=lambda whτ, wo, τ, *, ρ: np.divide(whτ, np.power(ρ + 1, np.divide(τ, 365))) + wo)
    wbo = Variable.Dependent("wbo", ("spot", Variables.Scenario.BREAKEVEN), np.float32, function=lambda wo, vlo: wo - vlo)
    wco = Variable.Dependent("wco", ("spot", Variables.Scenario.CURRENT), np.float32, function=lambda wo: wo)
    wlτ = Variable.Independent("wlτ", ("future", Variables.Scenario.MINIMUM), np.float32, locator="minimum")
    whτ = Variable.Independent("whτ", ("future", Variables.Scenario.MAXIMUM), np.float32, locator="maximum")

    def execute(self, *args, **kwargs):
        yield from super().execute(*args, **kwargs)
        yield self.vlo()
        yield self.vho()
        yield self.wlτ()
        yield self.whτ()
        yield self.wbo()
        yield self.wco()


class ExpectedEquation(ValuationEquation):
    veo = Variable.Dependent("veo", ("npv", Variables.Scenario.EXPECTED), np.float32, function=lambda weτ, wo, τ, *, ρ: np.divide(weτ, np.power(ρ + 1, np.divide(τ, 365))) + wo)
    weτ = Variable.Independent("weτ", ("future", Variables.Scenario.EXPECTED), np.float32, locator="expected")

    def execute(self, *args, **kwargs):
        yield from super().execute(*args, **kwargs)
        yield self.veo()
        yield self.weτ()


class GreekEquation(ValuationEquation):
    vo = Variable.Independent("vo", "value", np.float32, locator="value")
    Δo = Variable.Independent("Δo", "delta", np.float32, locator="delta")
    Γo = Variable.Independent("Γo", "gamma", np.float32, locator="gamma")
    Θo = Variable.Independent("Θo", "theta", np.float32, locator="theta")
    Vo = Variable.Independent("Vo", "vega", np.float32, locator="vega")
    Po = Variable.Independent("Po", "rho", np.float32, locator="rho")

    def execute(self, *args, **kwargs):
        yield from super().execute(*args, **kwargs)
        yield self.vo()
        yield self.Δo()
        yield self.Γo()
        yield self.Θo()
        yield self.Vo()
        yield self.Po()


class ValuationCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        required, optional = PayoffEquation, [ExpectedEquation, GreekEquation]
        calculation = Calculation[xr.DataArray](*args, required=required, optional=optional, **kwargs)
        self.__calculation = calculation

    def execute(self, strategies, *args, **kwargs):
        assert isinstance(strategies, (list, xr.Dataset))
        if self.empty(strategies, "size"): return
        settlements = self.keys(strategies, by=Querys.Settlement)
        settlements = ",".join(list(map(str, settlements)))
        valuations = self.calculate(strategies, *args, **kwargs)
        size = self.size(valuations)
        self.console(f"{str(settlements)}[{int(size):.0f}]")
        if self.empty(valuations): return
        yield valuations

    @TypeDispatcher(locator=0)
    def calculate(self, strategies, *args, **kwargs): raise TypeError(type(strategies))
    @calculate.register(list)
    def collection(self, strategies, *args, **kwargs):
        valuations = [self.calculate(dataset, *args, **kwargs) for dataset in strategies]
        valuations = pd.concat(valuations, axis=0)
        valuations = valuations.reset_index(drop=True, inplace=False)
        return valuations

    @calculate.register(xr.Dataset)
    def dataset(self, strategies, *args, **kwargs):
        print(strategies)

        valuations = self.calculation(strategies, *args, **kwargs)

        print(valuations)
        raise Exception()

        valuations = valuations.to_dataframe().dropna(how="all", inplace=False)
        valuations = valuations.reset_index(drop=False, inplace=False)
        options = [option for option in list(map(str, Securities.Options)) if option not in valuations.columns]
        for option in options: valuations[option] = np.NaN
        columns = {column: (column, "") for column in valuations.columns if not isinstance(column, tuple)}
        valuations = valuations.rename(columns=columns, inplace=False)
        valuations.columns = pd.MultiIndex.from_tuples(valuations.columns)
        valuations.columns.names = ["axis", "scenario"]
        return valuations

    @property
    def calculation(self): return self.__calculation



