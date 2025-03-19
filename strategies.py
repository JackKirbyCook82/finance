# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Strategy Objects
@author: Jack Kirby Cook

"""

import types
import numpy as np
import pandas as pd
import xarray as xr
from abc import ABC
from functools import reduce
from collections import namedtuple as ntuple

from finance.variables import Variables, Querys, Strategies, Securities
from support.calculations import Calculation, Equation, Variable
from support.mixins import Emptying, Sizing, Partition, Logging
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StrategyCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class StrategyLocator(ntuple("Locator", "axis position")): pass
class StrategyEquation(Equation, ABC):
    x = Variable("x", "underlying", np.float32, xr.DataArray, vectorize=True, function=lambda xα, xβ: (xα + xβ) / 2)
    q = Variable("q", "size", np.float32, xr.DataArray, vectorize=True, function=lambda qα, qβ: np.minimum(qα, qβ))
    w = Variable("w", "spot", np.float32, xr.DataArray, vectorize=True, function=lambda y, ε: y * 100 - ε)
    wh = Variable("wh", "maximum", np.float32, xr.DataArray, vectorize=True, function=lambda yh, ε: yh * 100 - ε)
    wl = Variable("wl", "minimum", np.float32, xr.DataArray, vectorize=True, function=lambda yl, ε: yl * 100 - ε)

    xα = Variable("xα", "underlying", np.float32, xr.DataArray, locator=StrategyLocator("underlying", Variables.Securities.Position.LONG))
    yα = Variable("yα", "price", np.float32, xr.DataArray, locator=StrategyLocator("price", Variables.Securities.Position.LONG))
    qα = Variable("qα", "size", np.float32, xr.DataArray, locator=StrategyLocator("size", Variables.Securities.Position.LONG))
    kα = Variable("kα", "strike", np.float32, xr.DataArray, locator=StrategyLocator("strike", Variables.Securities.Position.LONG))
    xβ = Variable("xβ", "underlying", np.float32, xr.DataArray, locator=StrategyLocator("underlying", Variables.Securities.Position.SHORT))
    yβ = Variable("yβ", "price", np.float32, xr.DataArray, locator=StrategyLocator("price", Variables.Securities.Position.SHORT))
    qβ = Variable("qβ", "size", np.float32, xr.DataArray, locator=StrategyLocator("size", Variables.Securities.Position.SHORT))
    kβ = Variable("kβ", "strike", np.float32, xr.DataArray, locator=StrategyLocator("strike", Variables.Securities.Position.SHORT))
    ε = Variable("ε", "fees", np.float32, types.NoneType, locator="fees")

class VerticalEquation(StrategyEquation):
    y = Variable("y", "spot", np.float32, xr.DataArray, vectorize=True, function=lambda yα, yβ: - yα + yβ)

class CollarEquation(StrategyEquation):
    y = Variable("y", "spot", np.float32, xr.DataArray, vectorize=True, function=lambda x, yα, yβ: - yα + yβ - (x / 100))

class VerticalPutEquation(VerticalEquation):
    yh = Variable("wh", "maximum", np.float32, xr.DataArray, vectorize=True, function=lambda kα, kβ: np.maximum(kα - kβ, 0))
    yl = Variable("wl", "minimum", np.float32, xr.DataArray, vectorize=True, function=lambda kα, kβ: np.minimum(kα - kβ, 0))

class VerticalCallEquation(VerticalEquation):
    yh = Variable("wh", "maximum", np.float32, xr.DataArray, vectorize=True, function=lambda kα, kβ: np.maximum(-kα + kβ, 0))
    yl = Variable("wl", "minimum", np.float32, xr.DataArray, vectorize=True, function=lambda kα, kβ: np.minimum(-kα + kβ, 0))

class CollarLongEquation(CollarEquation):
    yh = Variable("wh", "maximum", np.float32, xr.DataArray, vectorize=True, function=lambda kα, kβ: np.maximum(kα, kβ))
    yl = Variable("wl", "minimum", np.float32, xr.DataArray, vectorize=True, function=lambda kα, kβ: np.minimum(kα, kβ))

class CollarShortEquation(CollarEquation):
    yh = Variable("wh", "maximum", np.float32, xr.DataArray, vectorize=True, function=lambda kα, kβ: np.maximum(-kα, -kβ))
    yl = Variable("wl", "minimum", np.float32, xr.DataArray, vectorize=True, function=lambda kα, kβ: np.minimum(-kα, -kβ))


class StrategyCalculation(Calculation, ABC, metaclass=RegistryMeta):
    axes = ("price", "underlying", "strike", "size")

    def execute(self, securities, *args, fees, **kwargs):
        securities = {StrategyLocator(axis, security.position): dataset[axis] for security, dataset in securities.items() for axis in type(self).axes}
        with self.equation(securities, fees=fees) as equation:
            yield equation.q()
            yield equation.w()
            yield equation.wl()
            yield equation.wh()

class VerticalCalculation(StrategyCalculation, ABC): pass
class CollarCalculation(StrategyCalculation, ABC): pass
class VerticalPutCalculation(VerticalCalculation, equation=VerticalPutEquation, register=Strategies.Verticals.Put): pass
class VerticalCallCalculation(VerticalCalculation, equation=VerticalCallEquation, register=Strategies.Verticals.Call): pass
class CollarLongCalculation(CollarCalculation, equation=CollarLongEquation, register=Strategies.Collars.Long): pass
class CollarShortCalculation(CollarCalculation, equation=CollarShortEquation, register=Strategies.Collars.Short): pass


class StrategyCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, strategies=[], **kwargs):
        assert all([strategy in list(Strategies) for strategy in list(strategies)])
        super().__init__(*args, **kwargs)
        strategies = list(dict(StrategyCalculation).keys()) if not bool(strategies) else list(strategies)
        calculations = {strategy: calculation(*args, **kwargs) for strategy, calculation in dict(StrategyCalculation).items() if strategy in strategies}
        self.__calculations = calculations

    def execute(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if self.empty(options): return
        for settlement, dataframe in self.partition(options, by=Querys.Settlement):
            mapping = dict(self.options(dataframe, *args, **kwargs))
            strategies = self.calculate(mapping, *args, **kwargs)
            for strategy, dataset in strategies.items():
                size = self.size(dataset, "size")
                self.console(f"{str(settlement)}|{str(strategy)}[{int(size):.0f}]")
                if self.empty(dataset, "size"): continue
                yield dataset

    def calculate(self, options, *args, **kwargs):
        strategies = dict(self.calculator(options, *args, **kwargs))
        return strategies

    def calculator(self, options, *args, **kwargs):
        for strategy, calculation in self.calculations.items():
            if not all([option in options.keys() for option in list(strategy.options)]): continue
            securities = {security: options[security] for security in list(strategy.options)}
            strategies = calculation(securities, *args, **kwargs)
            assert isinstance(strategies, xr.Dataset)
            strategies = strategies.assign_coords({"strategy": xr.Variable("strategy", [strategy]).squeeze("strategy")})
            for field in list(Querys.Settlement): strategies = strategies.expand_dims(field)
            yield strategy, strategies

    def options(self, options, *args, **kwargs):
        for security, dataframe in options.groupby(list(Variables.Securities.Security), sort=False):
            if self.empty(dataframe): continue
            security = Securities(security)
            dataframe = dataframe.drop(columns=list(Variables.Securities.Security))
            dataframe = dataframe.set_index(list(Querys.Settlement) + ["strike"], drop=True, inplace=False)
            dataset = xr.Dataset.from_dataframe(dataframe)
            dataset = reduce(lambda content, axis: content.squeeze(axis), list(Querys.Settlement), dataset)
            dataset = dataset.rename({"strike": str(security)})
            dataset["strike"] = dataset[str(security)]
            yield security, dataset

    @property
    def calculations(self): return self.__calculations



