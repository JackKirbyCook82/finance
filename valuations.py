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
from calculations import Variables, Equations, Errors
from support.mixins import Emptying, Sizing, Partition, Logging
from support.decorators import Dispatchers

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

    def execute(self, strategies, /, current, discount, fees):
        yield from super().execute(strategies, current=current, discount=discount, fees=fees)
        yield self.qo(strategies, current=current, discount=discount, fees=fees)


class PayoffEquation(ValuationEquation):
    vl = Variables.Dependent("vl", "npv", np.float32, function=lambda wl, wo, τ, *, ρ: + np.divide(wl, np.power(ρ + 1, np.divide(τ, 365))) + wo)
    wk = Variables.Dependent("wk", "breakeven", np.float32, function=lambda wl, τ, *, ρ: - np.divide(wl, np.power(ρ + 1, np.divide(τ, 365))))
    wh = Variables.Dependent("wh", "maximum", np.float32, function=lambda yh, *, ε: yh * 100 - ε)
    wl = Variables.Dependent("wl", "minimum", np.float32, function=lambda yl, *, ε: yl * 100 - ε)
    wo = Variables.Dependent("wo", "spot", np.float32, function=lambda yo, *, ε: yo * 100 - ε)
    yh = Variables.Independent("yh", ("value", "maximum"), np.float32, locator="maximum")
    yl = Variables.Independent("yl", ("value", "minimum"), np.float32, locator="minimum")

    def execute(self, strategies, /, current, discount, fees):
        yield from super().execute(strategies, current=current, discount=discount, fees=fees)
        yield self.vl(strategies, current=current, discount=discount, fees=fees)
        yield self.wh(strategies, current=current, discount=discount, fees=fees)
        yield self.wk(strategies, current=current, discount=discount, fees=fees)
        yield self.wl(strategies, current=current, discount=discount, fees=fees)
        yield self.wo(strategies, current=current, discount=discount, fees=fees)


class UnderlyingEquation(ValuationEquation):
    xo = Variables.Independent("xo", "underlying", np.float32, locator="underlying")
    δo = Variables.Independent("δo", "volatility", np.float32, locator="volatility")
    μo = Variables.Independent("μo", "trend", np.float32, locator="trend")

    def execute(self, strategies, /, current, discount, fees):
        yield from super().execute(strategies, current=current, discount=discount, fees=fees)
        for attribute in str("λα,λβ").split(","):
            try: content = getattr(self, attribute)(strategies, current=current, discount=discount, fees=fees)
            except Errors.Independent: continue
            yield content


class AppraisalEquation(ValuationEquation):
    vo = Variables.Independent("vo", "value", np.float32, locator="value")
    Δo = Variables.Independent("Δo", "delta", np.float32, locator="delta")
    Γo = Variables.Independent("Γo", "gamma", np.float32, locator="gamma")
    Θo = Variables.Independent("Θo", "theta", np.float32, locator="theta")
    Vo = Variables.Independent("Vo", "vega", np.float32, locator="vega")
#    λα = Variables.Independent("λα", "buying", np.float32, locator="buying")
#    λβ = Variables.Independent("λβ", "selling", np.float32, locator="selling")

    def execute(self, strategies, /, current, discount, fees):
        yield from super().execute(strategies, current=current, discount=discount, fees=fees)
        for attribute in str("vo,Δo,Γo,Θo,Vo").split(","):
            try: content = getattr(self, attribute)(strategies, current=current, discount=discount, fees=fees)
            except Errors.Independent: continue
            yield content


class ValuationCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        equation = ValuationEquation + list(ValuationEquation.__subclasses__())
        self.__equation = equation(*args, **kwargs)

    def execute(self, strategies, /, **kwargs):
        assert isinstance(strategies, (list, xr.Dataset))
        if self.empty(strategies, "size"): return
        generator = self.calculator(strategies, **kwargs)
        for settlement, valuations in generator:
            size = self.size(valuations)
            self.console(f"{str(settlement)}[{int(size):.0f}]")
            if self.empty(valuations): return
            yield valuations

    def calculator(self, strategies, *args, **kwargs):
        for settlement, datasets in self.partition(strategies, by=Querys.Settlement):
            valuations = self.calculate(datasets, *args, **kwargs)
            yield settlement, valuations

    @Dispatchers.Type(locator=0)
    def calculate(self, strategies, *args, **kwargs): raise TypeError(type(strategies))
    @calculate.register(list)
    def collection(self, strategies, *args, **kwargs):
        valuations = [self.calculate(dataset, *args, **kwargs) for dataset in strategies]
        valuations = pd.concat(valuations, axis=0)
        valuations = valuations.reset_index(drop=True, inplace=False)
        return valuations

    @calculate.register(xr.Dataset)
    def dataset(self, strategies, *args, current, discount, interest, fees, **kwargs):
        valuations = self.equation(strategies, current=current, discount=discount, interest=interest, fees=fees)
        valuations = valuations.to_dataframe().dropna(how="all", inplace=False)
        valuations = valuations.reset_index(drop=False, inplace=False)
        options = [option for option in list(map(str, Securities.Options)) if option not in valuations.columns]
        for option in options: valuations[option] = np.NaN
        return valuations

    @property
    def equation(self): return self.__equation



