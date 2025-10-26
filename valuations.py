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
from calculations import Equation, Variables, Algorithms, Computations, Errors
from support.mixins import Emptying, Sizing, Partition, Logging
from support.decorators import Dispatchers

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ValuationCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class ValuationEquation(Computations.Array, Algorithms.Vectorized.Array, Equation, ABC, root=True):
    τ = Variables.Dependent("τ", "tau", np.int32, function=lambda tτ, *, to: (tτ - to).days)

    wo = Variables.Independent("wo", ("value", "spot"), np.float32, locator="spot")
    qo = Variables.Independent("qo", "size", np.float32, locator="size")
    tτ = Variables.Independent("tτ", "expire", Date, locator="expire")
    to = Variables.Constant("to", "current", Date, locator="current")

    r = Variables.Constant("r", "interest", np.float32, locator="interest")
    ρ = Variables.Constant("ρ", "discount", np.float32, locator="discount")
    ε = Variables.Constant("ε", "fees", np.float32, locator="fees")

    def execute(self, strategies, /, current, interest, discount, fees):
        parameters = dict(current=current, interest=interest, discount=discount, fees=fees)
        yield from super().execute(strategies, **parameters)
        yield self.qo(strategies, **parameters)


class PayoffEquation(ValuationEquation, ABC):
    vl = Variables.Dependent("vl", "npv", np.float32, function=lambda ul, uo, τ, *, ρ: + np.divide(ul, np.power(ρ + 1, np.divide(τ, 365))) + uo)
    uk = Variables.Dependent("uk", "breakeven", np.float32, function=lambda ul, τ, *, ρ: - np.divide(ul, np.power(ρ + 1, np.divide(τ, 365))))
    uh = Variables.Dependent("uh", "maximum", np.float32, function=lambda wh, *, ε: wh * 100 - ε)
    ul = Variables.Dependent("ul", "minimum", np.float32, function=lambda wl, *, ε: wl * 100 - ε)
    uo = Variables.Dependent("uo", "spot", np.float32, function=lambda wo, *, ε: wo * 100 - ε)
    wh = Variables.Independent("wh", ("value", "maximum"), np.float32, locator="maximum")
    wl = Variables.Independent("wl", ("value", "minimum"), np.float32, locator="minimum")

    def execute(self, strategies, /, current, interest, discount, fees):
        parameters = dict(current=current, interest=interest, discount=discount, fees=fees)
        yield from super().execute(strategies, **parameters)
        yield self.vl(strategies, **parameters)
        yield self.wh(strategies, **parameters)
        yield self.wk(strategies, **parameters)
        yield self.wl(strategies, **parameters)
        yield self.wo(strategies, **parameters)


class UnderlyingEquation(ValuationEquation, ABC):
    xo = Variables.Independent("xo", "underlying", np.float32, locator="underlying")
    δo = Variables.Independent("δo", "volatility", np.float32, locator="volatility")
    μo = Variables.Independent("μo", "trend", np.float32, locator="trend")

    def execute(self, strategies, /, current, interest, discount, fees):
        parameters = dict(current=current, interest=interest, discount=discount, fees=fees)
        yield from super().execute(strategies, **parameters)
        for attribute in str("xo,μo,δo").split(","):
            try: content = getattr(self, attribute)(strategies, **parameters)
            except Errors.Independent: continue
            yield content


class AppraisalEquation(ValuationEquation, ABC):
    yo = Variables.Independent("yo", "value", np.float32, locator="value")
    Δo = Variables.Independent("Δo", "delta", np.float32, locator="delta")
    Γo = Variables.Independent("Γo", "gamma", np.float32, locator="gamma")
    Θo = Variables.Independent("Θo", "theta", np.float32, locator="theta")
    Vo = Variables.Independent("Vo", "vega", np.float32, locator="vega")
#    λα = Variables.Independent("λα", "buying", np.float32, locator="buying")
#    λβ = Variables.Independent("λβ", "selling", np.float32, locator="selling")

    def execute(self, strategies, /, current, interest, discount, fees):
        yield from super().execute(strategies, current=current, interest=interest, discount=discount, fees=fees)
        for attribute in str("yo,Δo,Γo,Θo,Vo").split(","):
            try: content = getattr(self, attribute)(strategies, current=current, interest=interest, discount=discount, fees=fees)
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



