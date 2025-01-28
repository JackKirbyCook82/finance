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
from itertools import product
from collections import namedtuple as ntuple

from finance.variables import Variables, Querys, Strategies, Securities
from support.calculations import Calculation, Equation, Variable
from support.mixins import Emptying, Sizing, Partition
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StrategyCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class StrategyLocator(ntuple("Locator", "axis position")): pass
class StrategyEquation(Equation, ABC):
    t = Variable("t", "current", np.datetime64, xr.DataArray, vectorize=True, function=lambda tα, tβ: np.minimum(np.datetime64(tα, "ns"), np.datetime64(tβ, "ns")))
    x = Variable("x", "underlying", np.float32, xr.DataArray, vectorize=True, function=lambda xα, xβ: (xα + xβ) / 2)
    w = Variable("w", "spot", np.float32, xr.DataArray, vectorize=True, function=lambda y, ε: y * 100 - ε)
    q = Variable("q", "size", np.float32, xr.DataArray, vectorize=True, function=lambda qα, qβ: np.minimum(qα, qβ))
    wh = Variable("wh", "maximum", np.float32, xr.DataArray, vectorize=True, function=lambda yh, ε: yh * 100 - ε)
    wl = Variable("wl", "minimum", np.float32, xr.DataArray, vectorize=True, function=lambda yl, ε: yl * 100 - ε)

    tα = Variable("tα", "current", np.datetime64, xr.DataArray, locator=StrategyLocator("current", Variables.Securities.Position.LONG))
    xα = Variable("xα", "underlying", np.float32, xr.DataArray, locator=StrategyLocator("underlying", Variables.Securities.Position.LONG))
    yα = Variable("yα", "ask", np.float32, xr.DataArray, locator=StrategyLocator("ask", Variables.Securities.Position.LONG))
    qα = Variable("qα", "supply", np.float32, xr.DataArray, locator=StrategyLocator("supply", Variables.Securities.Position.LONG))
    kα = Variable("kα", "strike", np.float32, xr.DataArray, locator=StrategyLocator("strike", Variables.Securities.Position.LONG))
    tβ = Variable("tβ", "current", np.datetime64, xr.DataArray, locator=StrategyLocator("current", Variables.Securities.Position.SHORT))
    xβ = Variable("xβ", "underlying", np.float32, xr.DataArray, locator=StrategyLocator("underlying", Variables.Securities.Position.SHORT))
    yβ = Variable("yβ", "bid", np.float32, xr.DataArray, locator=StrategyLocator("bid", Variables.Securities.Position.SHORT))
    qβ = Variable("qβ", "demand", np.float32, xr.DataArray, locator=StrategyLocator("demand", Variables.Securities.Position.SHORT))
    kβ = Variable("kβ", "strike", np.float32, xr.DataArray, locator=StrategyLocator("strike", Variables.Securities.Position.SHORT))
    ε = Variable("ε", "fees", np.float32, types.NoneType, locator="fees")

class VerticalEquation(StrategyEquation):
    y = Variable("y", "spot", np.float32, xr.DataArray, vectorize=True, function=lambda yα, yβ: - yα + yβ)

class CollarEquation(StrategyEquation):
    y = Variable("y", "spot", np.float32, xr.DataArray, vectorize=True, function=lambda x, yα, yβ: - yα + yβ - x)

class VerticalPutEquation(VerticalEquation):
    yh = Variable("yh", "maximum", np.float32, xr.DataArray, vectorize=True, function=lambda kα, kβ: np.maximum(kα - kβ, 0))
    yl = Variable("yl", "minimum", np.float32, xr.DataArray, vectorize=True, function=lambda kα, kβ: np.minimum(kα - kβ, 0))

class VerticalCallEquation(VerticalEquation):
    yh = Variable("yh", "maximum", np.float32, xr.DataArray, vectorize=True, function=lambda kα, kβ: np.maximum(-kα + kβ, 0))
    yl = Variable("yl", "minimum", np.float32, xr.DataArray, vectorize=True, function=lambda kα, kβ: np.minimum(-kα + kβ, 0))

class CollarLongEquation(CollarEquation):
    yh = Variable("yh", "maximum", np.float32, xr.DataArray, vectorize=True, function=lambda kα, kβ: np.maximum(kα, kβ))
    yl = Variable("yl", "minimum", np.float32, xr.DataArray, vectorize=True, function=lambda kα, kβ: np.minimum(kα, kβ))

class CollarShortEquation(CollarEquation):
    yh = Variable("yh", "maximum", np.float32, xr.DataArray, vectorize=True, function=lambda kα, kβ: np.maximum(-kα, -kβ))
    yl = Variable("yl", "minimum", np.float32, xr.DataArray, vectorize=True, function=lambda kα, kβ: np.minimum(-kα, -kβ))


class StrategyCalculation(Calculation, ABC, metaclass=RegistryMeta):
    axes = ("price", "underlying", "strike", "size", "current")

    def execute(self, securities, *args, fees, **kwargs):
        securities = {StrategyLocator(axis, security.position): dataset[axis] for security, dataset in securities.items() for axis in type(self).axes}
        with self.equation(securities, fees=fees) as equation:
            yield equation.t()
            yield equation.q()
            yield equation.x()
            yield equation.y()
            yield equation.yl()
            yield equation.yh()

class VerticalCalculation(StrategyCalculation, ABC): pass
class CollarCalculation(StrategyCalculation, ABC): pass
class VerticalPutCalculation(VerticalCalculation, equation=VerticalPutEquation, register=Strategies.Verticals.Put): pass
class VerticalCallCalculation(VerticalCalculation, equation=VerticalCallEquation, register=Strategies.Verticals.Call): pass
class CollarLongCalculation(CollarCalculation, equation=CollarLongEquation, register=Strategies.Collars.Long): pass
class CollarShortCalculation(CollarCalculation, equation=CollarShortEquation, register=Strategies.Collars.Short): pass


class StrategyCalculator(Sizing, Emptying, Partition, query=Querys.Settlement, title="Calculated"):
    def __init__(self, *args, strategies=[], **kwargs):
        assert all([strategy in list(Strategies) for strategy in list(strategies)])
        super().__init__(*args, **kwargs)
        strategies = list(dict(StrategyCalculation).keys()) if not bool(strategies) else list(strategies)
        calculations = {strategy: calculation(*args, **kwargs) for strategy, calculation in dict(StrategyCalculation).items() if strategy in strategies}
        self.__calculations = calculations

    def execute(self, securities, *args, **kwargs):
        assert isinstance(securities, pd.DataFrame)
        if self.empty(securities): return
        for settlement, dataframe in self.partition(securities):
            contents = dict(self.securities(dataframe, *args, **kwargs))
            for strategy, strategies in self.calculator(contents, *args, **kwargs):
                size = self.size(strategies)
                string = f"{str(settlement)}|{str(strategy)}[{int(size):.0f}]"
                self.console(string)
                if self.empty(strategies, "size"): continue
                yield strategies

    def calculator(self, securities, *args, **kwargs):
        for strategy, calculation in self.calculations.items():
            if not all([security in securities.keys() for security in list(strategy.options)]): continue
            securities = {security: securities[security] for security in list(strategy.options)}
            strategies = calculation(securities, *args, **kwargs)
            assert isinstance(strategies, xr.Dataset)
            strategies = strategies.assign_coords({"strategy": xr.Variable("strategy", [strategy]).squeeze("strategy")})
            for field in list(Querys.Settlement): strategies = strategies.expand_dims(field)
            yield strategy, strategies

    def securities(self, options, *args, **kwargs):
        options, positions = options.groupby("option", sort=False), list(Variables.Securities.Position)
        for position, (option, dataframe) in product(options, positions):
            if self.empty(dataframe): continue
            security = Securities([Variables.Securities.Instrument.OPTION, option, position])
            dataframe = dataframe.dropna("option", axis=1)
            dataframe = dataframe.set_index(list(Querys.Settlement) + ["strike"], drop=True, inplace=False)
            dataset = xr.Dataset.from_dataframe(dataframe)
            dataset = reduce(lambda content, axis: content.squeeze(axis), list(Querys.Settlement), dataset)
            dataset = dataset.rename({"strike": str(security)})
            dataset["strike"] = dataset[str(security)]
            yield security, dataset

    @property
    def calculations(self): return self.__calculations



