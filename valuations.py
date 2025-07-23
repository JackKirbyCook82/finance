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

from finance.variables import Querys, Securities
from support.calculations import Calculation, Equation, Variable
from support.mixins import Emptying, Sizing, Partition, Logging
from support.decorators import TypeDispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ValuationCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class ValuationEquation(Equation, ABC, datatype=xr.DataArray, vectorize=True):
    τ = Variable.Dependent("τ", "tau", np.int32, function=lambda tτ, *, to: (tτ - to).days)

    yh = Variable.Independent("yh", "maximum", np.float32, locator="maximum")
    yl = Variable.Independent("yl", "minimum", np.float32, locator="minimum")
    yo = Variable.Independent("yo", "spot", np.float32, locator="spot")
    qo = Variable.Independent("qo", "size", np.float32, locator="size")
    xo = Variable.Independent("xo", "underlying", np.float32, locator="underlying")
    μo = Variable.Independent("μo", "trend", np.float32, locator="trend")
    δo = Variable.Independent("δo", "volatility", np.float32, locator="volatility")
    tτ = Variable.Independent("tτ", "expire", Date, locator="expire")
    to = Variable.Constant("to", "current", Date, locator="current")

    r = Variable.Constant("r", "interest", np.float32, locator="interest")
    q = Variable.Constant("q", "dividend", np.float32, locator="dividend")
    ε = Variable.Constant("ε", "fees", np.float32, locator="fees")

    def execute(self, *args, **kwargs):
        yield from super().execute(*args, **kwargs)
        yield self.yh()
        yield self.yo()
        yield self.yl()


class GreeksEquation(ValuationEquation):
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
        calculation = Calculation[xr.DataArray](*args, required=ValuationEquation, optional=[GreeksEquation], **kwargs)
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
    def dataset(self, strategies, *args, current, discount, interest, dividend, fees, **kwargs):
        parameters = dict(current=current, discount=discount, interest=interest, dividend=dividend, fees=fees)
        valuations = self.calculation(strategies, *args, **parameters, **kwargs)
        valuations = valuations.to_dataframe().dropna(how="all", inplace=False)
        valuations = valuations.reset_index(drop=False, inplace=False)
        options = [option for option in list(map(str, Securities.Options)) if option not in valuations.columns]
        for option in options: valuations[option] = np.NaN
        return valuations

    @property
    def calculation(self): return self.__calculation



