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

from finance.concepts import Querys, Securities
from support.mixins import Emptying, Sizing, Partition, Logging
from support.decorators import TypeDispatcher
from calculations import Variables, Equations, DomainError

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ValuationCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class ValuationEquation(Equations.Vectorized.Array, ABC):
    τ = Variables.Dependent("τ", "tau", np.int32, function=lambda tτ, *, to: (tτ - to).days)

    yo = Variables.Independent("yo", ("value", "spot"), np.float32, locator="spot")
    qo = Variables.Independent("qo", "size", np.float32, locator="size")
    tτ = Variables.Independent("tτ", "expire", Date, locator="expire")
    to = Variables.Constant("to", "current", Date, locator="current")

    r = Variables.Constant("r", "interest", np.float32, locator="interest")
    ρ = Variables.Constant("ρ", "discount", np.float32, locator="discount")
    ε = Variables.Constant("ε", "fees", np.float32, locator="fees")

    def execute(self, *args, **kwargs):
        yield from super().execute(*args, **kwargs)
        yield self.qo()


class PayoffEquation(ValuationEquation):
    vl = Variables.Dependent("vl", "npv", np.float32, function=lambda wl, wo, τ, *, ρ: + np.divide(wl, np.power(ρ + 1, np.divide(τ, 365))) + wo)
    wk = Variables.Dependent("wk", "breakeven", np.float32, function=lambda wl, τ, *, ρ: - np.divide(wl, np.power(ρ + 1, np.divide(τ, 365))))
    wh = Variables.Dependent("wh", "maximum", np.float32, function=lambda yh, *, ε: yh * 100 - ε)
    wl = Variables.Dependent("wl", "minimum", np.float32, function=lambda yl, *, ε: yl * 100 - ε)
    wo = Variables.Dependent("wo", "spot", np.float32, function=lambda yo, *, ε: yo * 100 - ε)
    yh = Variables.Independent("yh", ("value", "maximum"), np.float32, locator="maximum")
    yl = Variables.Independent("yl", ("value", "minimum"), np.float32, locator="minimum")

    def execute(self, *args, **kwargs):
        yield from super().execute(*args, **kwargs)
        yield self.vl()
        yield self.wh()
        yield self.wk()
        yield self.wl()
        yield self.wo()

class UnderlyingEquation(ValuationEquation):
    xo = Variables.Independent("xo", "underlying", np.float32, locator="underlying")
    δo = Variables.Independent("δo", "volatility", np.float32, locator="volatility")
    μo = Variables.Independent("μo", "trend", np.float32, locator="trend")

    def execute(self, *args, **kwargs):
        yield from super().execute(*args, **kwargs)
        for attribute in str("xo,δo,μo").split(","):
            try: getattr(self, attribute)()
            except DomainError: pass

class AppraisalEquation(ValuationEquation):
    vo = Variables.Independent("vo", "value", np.float32, locator="value")
    Δo = Variables.Independent("Δo", "delta", np.float32, locator="delta")
    Γo = Variables.Independent("Γo", "gamma", np.float32, locator="gamma")
    Θo = Variables.Independent("Θo", "theta", np.float32, locator="theta")
    Vo = Variables.Independent("Vo", "vega", np.float32, locator="vega")

    def execute(self, *args, **kwargs):
        yield from super().execute(*args, **kwargs)
        for attribute in str("vo,Δo,Γo,Θo,Vo").split(","):
            try: getattr(self, attribute)()
            except DomainError: pass


class ValuationCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__equation = ValuationEquation + list(ValuationEquation.__subclasses__())

    def execute(self, strategies, *args, **kwargs):
        assert isinstance(strategies, (list, xr.Dataset))
        if self.empty(strategies, "size"): return
        generator = self.calculator(strategies, *args, **kwargs)
        for settlement, valuations in generator:
            size = self.size(valuations)
            self.console(f"{str(settlement)}[{int(size):.0f}]")
            if self.empty(valuations): return
            yield valuations

    def calculator(self, strategies, *args, **kwargs):
        for settlement, datasets in self.partition(strategies, by=Querys.Settlement):
            valuations = self.calculate(datasets, *args, **kwargs)
            yield settlement, valuations

    @TypeDispatcher(locator=0)
    def calculate(self, strategies, *args, **kwargs): raise TypeError(type(strategies))
    @calculate.register(list)
    def collection(self, strategies, *args, **kwargs):
        valuations = [self.calculate(dataset, *args, **kwargs) for dataset in strategies]
        valuations = pd.concat(valuations, axis=0)
        valuations = valuations.reset_index(drop=True, inplace=False)
        return valuations

    @calculate.register(xr.Dataset)
    def dataset(self, strategies, *args, current, discount, interest, fees, **kwargs):
        parameters = dict(current=current, discount=discount, interest=interest, fees=fees)
        equation = self.equation(arguments=strategies, parameters=parameters)
        valuations = equation(*args, **kwargs)
        valuations = valuations.to_dataframe().dropna(how="all", inplace=False)
        valuations = valuations.reset_index(drop=False, inplace=False)
        options = [option for option in list(map(str, Securities.Options)) if option not in valuations.columns]
        for option in options: valuations[option] = np.NaN
        return valuations

    @property
    def equation(self): return self.__equation



