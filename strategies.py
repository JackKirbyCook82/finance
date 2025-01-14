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
from itertools import product
from functools import reduce
from collections import namedtuple as ntuple

from finance.variables import Variables, Categories, Querys
from support.mixins import Emptying, Sizing, Logging, Segregating
from support.calculations import Calculation, Equation, Variable
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StrategyCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class StrategyLocator(ntuple("Source", "axis position")): pass
class StrategyEquation(Equation, ABC):
    t = Variable("t", "current", np.datetime64, xr.DataArray, vectorize=True, function=lambda tα, tβ: np.minimum(np.datetime64(tα, "ns"), np.datetime64(tβ, "ns")))
    q = Variable("q", "size", np.float32, xr.DataArray, vectorize=True, function=lambda qα, qβ: np.minimum(qα, qβ))
    x = Variable("x", "underlying", np.float32, xr.DataArray, vectorize=True, function=lambda xα, xβ: (xα + xβ) / 2)
    w = Variable("w", "spot", np.float32, xr.DataArray, vectorize=True, function=lambda y, ε: y * 100 - ε)
    wh = Variable("wh", "maximum", np.float32, xr.DataArray, vectorize=True, function=lambda yh, ε: yh * 100 - ε)
    wl = Variable("wl", "minimum", np.float32, xr.DataArray, vectorize=True, function=lambda yl, ε: yl * 100 - ε)

    tα = Variable("tα", "current", np.datetime64, xr.DataArray, locator=StrategyLocator("current", Variables.Positions.LONG))
    qα = Variable("qα", "size", np.float32, xr.DataArray, locator=StrategyLocator("size", Variables.Positions.LONG))
    xα = Variable("xα", "underlying", np.float32, xr.DataArray, locator=StrategyLocator("underlying", Variables.Positions.LONG))
    yα = Variable("yα", "price", np.float32, xr.DataArray, locator=StrategyLocator("price", Variables.Positions.LONG))
    kα = Variable("kα", "strike", np.float32, xr.DataArray, locator=StrategyLocator("strike", Variables.Positions.LONG))
    tβ = Variable("tβ", "current", np.datetime64, xr.DataArray, locator=StrategyLocator("current", Variables.Positions.SHORT))
    qβ = Variable("qβ", "size", np.float32, xr.DataArray, locator=StrategyLocator("size", Variables.Positions.SHORT))
    xβ = Variable("xβ", "underlying", np.float32, xr.DataArray, locator=StrategyLocator("underlying", Variables.Positions.SHORT))
    yβ = Variable("yβ", "price", np.float32, xr.DataArray, locator=StrategyLocator("price", Variables.Positions.SHORT))
    kβ = Variable("kβ", "strike", np.float32, xr.DataArray, locator=StrategyLocator("strike", Variables.Positions.SHORT))
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


class StrategyCalculation(Calculation, metaclass=RegistryMeta):
    def execute(self, options, *args, fees, **kwargs):
        positions = [option.position for option in options.keys()]
        assert len(set(positions)) == len(list(positions))
        options = {option.position: dataset for option, dataset in options.items()}
        variables, positions = ("price", "strike", "size", "underlying", "current"), list(Variables.Positions)
        sources = {StrategyLocator(variable, position): options[position][variable] for variable, position in product(variables, positions)}
        with self.equation(sources, fees=fees) as equation:
            yield equation.t()
            yield equation.q()
            yield equation.x()
            yield equation.y()
            yield equation.yl()
            yield equation.yh()


class VerticalCalculation(StrategyCalculation): pass
class CollarCalculation(StrategyCalculation): pass
class VerticalPutCalculation(VerticalCalculation, equation=VerticalPutEquation, register=Categories.Strategies.Verticals.Put): pass
class VerticalCallCalculation(VerticalCalculation, equation=VerticalCallEquation, register=Categories.Strategies.Verticals.Call): pass
class CollarLongCalculation(CollarCalculation, equation=CollarLongEquation, register=Categories.Strategies.Collars.Long): pass
class CollarShortCalculation(CollarCalculation, equation=CollarShortEquation, register=Categories.Strategies.Collars.Short): pass


class StrategyCalculator(Segregating, Sizing, Emptying, Logging):
    def __init__(self, *args, strategies=[], **kwargs):
        assert all([strategy in list(Categories.Strategies) for strategy in list(strategies)])
        super().__init__(*args, **kwargs)
        strategies = list(dict(StrategyCalculation).keys()) if not bool(strategies) else list(strategies)
        calculations = dict(StrategyCalculation).items()
        calculations = {strategy: calculation(*args, **kwargs) for strategy, calculation in calculations if strategy in strategies}
        self.__calculations = calculations

    def execute(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if self.empty(options): return
        for query, dataframe in self.segregate(options, *args, **kwargs):
            contents = dict(self.options(dataframe, *args, **kwargs))
            for strategy, strategies in self.calculate(contents, *args, **kwargs):
                size = self.size(strategies, "size")
                string = f"Calculated: {repr(self)}|{str(query)}|{str(strategy)}[{size:.0f}]"
                self.logger.info(string)
                if self.empty(strategies, "size"): continue
                yield strategies

    def options(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        for security, dataframe in options.groupby(list(Variables.Security), sort=False):
            if self.empty(dataframe): continue
            security = Categories.Securities(security)
            dataframe = dataframe.set_index(list(Querys.Product), drop=True, inplace=False)
            columns = [column for column in list(Variables.Security) if column in dataframe.columns]
            dataframe = dataframe.drop(columns, axis=1)
            dataset = xr.Dataset.from_dataframe(dataframe)
            dataset = reduce(lambda content, axis: content.squeeze(axis), list(Querys.Contract), dataset)
            dataset = dataset.drop_vars(list(Variables.Security), errors="ignore")
            dataset = dataset.rename({"strike": str(security)})
            dataset["strike"] = dataset[str(security)]
            yield security, dataset

    def calculate(self, options, *args, **kwargs):
        assert isinstance(options, dict)
        function = lambda mapping: {key: xr.Variable(key, [value]).squeeze(key) for key, value in mapping.items()}
        for strategy, calculation in self.calculations.items():
            if not all([security in options.keys() for security in list(strategy.options)]): continue
            datasets = {option: options[option] for option in list(strategy.options)}
            strategies = calculation(datasets, *args, **kwargs)
            assert isinstance(strategies, xr.Dataset)
            coordinates = function(dict(strategy=strategy))
            strategies = strategies.assign_coords(coordinates)
            for field in list(Querys.Contract): strategies = strategies.expand_dims(field)
            yield strategy, strategies

    @property
    def calculations(self): return self.__calculations



