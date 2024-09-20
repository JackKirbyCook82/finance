# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Strategy Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd
import xarray as xr
from abc import ABC
from functools import reduce
from itertools import product
from collections import OrderedDict as ODict

from finance.variables import Variables, Contract
from support.calculations import Variable, Equation, Calculation
from support.processes import Calculator
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StrategyCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class StrategyEquation(Equation):
    t = Variable("t", "current", np.datetime64, function=lambda tα, tβ: np.minimum(np.datetime64(tα, "ns"), np.datetime64(tβ, "ns")))
    q = Variable("q", "size", np.float32, function=lambda qα, qβ: np.minimum(qα, qβ))
    x = Variable("x", "underlying", np.float32, function=lambda xα, xβ: (xα + xβ) / 2)
    w = Variable("w", "spot", np.float32, function=lambda y, ε: y * 100 - ε)
    wh = Variable("wh", "maximum", np.float32, function=lambda yh, ε: yh * 100 - ε)
    wl = Variable("wl", "minimum", np.float32, function=lambda yl, ε: yl * 100 - ε)
    tα = Variable("tα", "current", np.datetime64, position=Variables.Positions.LONG, locator="current")
    qα = Variable("qα", "size", np.float32, position=Variables.Positions.LONG, locator="size")
    xα = Variable("xα", "underlying", np.float32, position=Variables.Positions.LONG, locator="underlying")
    yα = Variable("yα", "price", np.float32, position=Variables.Positions.LONG, locator="price")
    kα = Variable("kα", "strike", np.float32, position=Variables.Positions.LONG, locator="strike")
    tβ = Variable("tβ", "current", np.datetime64, position=Variables.Positions.SHORT, locator="current")
    qβ = Variable("qβ", "size", np.float32, position=Variables.Positions.SHORT, locator="size")
    xβ = Variable("xβ", "underlying", np.float32, position=Variables.Positions.SHORT, locator="underlying")
    yβ = Variable("yβ", "price", np.float32, position=Variables.Positions.SHORT, locator="price")
    kβ = Variable("kβ", "strike", np.float32, position=Variables.Positions.SHORT, locator="strike")
    ε = Variable("ε", "fees", np.float32, position="fees")

class VerticalEquation(StrategyEquation):
    y = Variable("y", "spot", np.float32, function=lambda yα, yβ: - yα + yβ)

class CollarEquation(StrategyEquation):
    y = Variable("y", "spot", np.float32, function=lambda yα, yβ, x: - yα + yβ - x)

class VerticalPutEquation(VerticalEquation):
    yh = Variable("yh", "maximum", np.float32, function=lambda kα, kβ: np.maximum(kα - kβ, 0))
    yl = Variable("yl", "minimum", np.float32, function=lambda kα, kβ: np.minimum(kα - kβ, 0))

class VerticalCallEquation(VerticalEquation):
    yh = Variable("yh", "maximum", np.float32, function=lambda kα, kβ: np.maximum(-kα + kβ, 0))
    yl = Variable("yl", "minimum", np.float32, function=lambda kα, kβ: np.minimum(-kα + kβ, 0))

class CollarLongEquation(CollarEquation):
    yh = Variable("yh", "maximum", np.float32, function=lambda kα, kβ: np.maximum(kα, kβ))
    yl = Variable("yl", "minimum", np.float32, function=lambda kα, kβ: np.minimum(kα, kβ))

class CollarShortEquation(CollarEquation):
    yh = Variable("yh", "maximum", np.float32, function=lambda kα, kβ: np.maximum(-kα, -kβ))
    yl = Variable("yl", "minimum", np.float32, function=lambda kα, kβ: np.minimum(-kα, -kβ))


class StrategyCalculation(Calculation, ABC, register=RegistryMeta):
    def execute(self, options, *args, fees, **kwargs):
        positions = [option.position for option in options.keys()]
        assert len(set(positions)) == len(list(positions))
        options = {option.position: dataset for option, dataset in options.items()}
        equation = self.equation(*args, **kwargs)
        yield equation.wh(options, fees=fees)
        yield equation.wl(options, fees=fees)
        yield equation.w(options, fees=fees)
        yield equation.x(options, fees=fees)
        yield equation.q(options, fees=fees)
        yield equation.t(options, fees=fees)

class VerticalPutCalculation(StrategyCalculation, equation=VerticalPutEquation, regsiter=Variables.Strategies.Vertical.Put): pass
class VerticalCallCalculation(StrategyCalculation, equation=VerticalCallEquation, regsiter=Variables.Strategies.Vertical.Call): pass
class CollarLongCalculation(StrategyCalculation, equation=CollarLongEquation, regsiter=Variables.Strategies.Collar.Long): pass
class CollarShortCalculation(StrategyCalculation, equation=CollarShortEquation, regsiter=Variables.Strategies.Collar.Short): pass


class StrategyVariables(object):
    axes = {Variables.Querys.CONTRACT: ["ticker", "expire"], Variables.Datasets.SECURITY: ["instrument", "option", "position"]}

    def __init__(self, *args, strategies=[], **kwargs):
        assert isinstance(strategies, list) and all([strategy in list(Variables.Strategies) for strategy in strategies])
        self.index = self.axes[Variables.Querys.CONTRACT] + self.axes[Variables.Datasets.SECURITY] + ["strike"]
        self.strategies = list(strategies) if bool(strategies) else list(Variables.Strategies)
        self.security = self.axes[Variables.Datasets.SECURITY]
        self.contract = self.axes[Variables.Querys.CONTRACT]


class StrategyCalculator(Calculator, calculations=dict(StrategyCalculation), variables=StrategyVariables):
    def execute(self, contract, options, *args, **kwargs):
        assert isinstance(contract, Contract) and isinstance(options, pd.DataFrame)
        options = ODict(list(self.options(options, *args, **kwargs)))
        strategies = ODict(list(self.strategies(options, *args, **kwargs)))
        strategies = list(strategies.values())
        assert all([isinstance(dataset, xr.Dataset) for dataset in strategies])
        size = self.size([dataset["size"] for dataset in strategies.values()])
        string = f"{str(self.title)}: {repr(self)}|{str(contract)}[{size:.0f}]"
        self.logger.info(string)
        return strategies

    def strategies(self, options, *args, **kwargs):
        if bool(options.empty): return
        function = lambda mapping: {key: xr.Variable(key, [value]).squeeze(key) for key, value in mapping.items()}
        for strategy, calculation in self.calculations.items():
            if strategy not in self.variables.strategies: continue
            if not all([option in options.keys() for option in list(strategy.options)]): continue
            datasets = {option: options[option] for option in list(strategy.options)}
            strategies = calculation(datasets, *args, **kwargs)
            assert isinstance(strategies, xr.Dataset)
            if self.empty(strategies["size"]): continue
            coordinates = function(dict(strategy=strategy))
            strategies = strategies.assign_coords(coordinates)
            yield strategy, strategies

    def options(self, options, *args, **kwargs):
        if bool(options.empty): return
        options = options.set_index(self.variables.index, drop=True, inplace=False)
        options = xr.Dataset.from_dataframe(options)
        options = reduce(lambda content, axis: content.squeeze(axis), self.variables.contract, options)
        for values in product(*[options[axis].values for axis in self.variables.security]):
            dataset = options.sel(indexers={key: value for key, value in zip(self.variables.security, values)})
            dataset = dataset.drop_vars(self.variables.security, errors="ignore")
            if self.empty(dataset["size"]): continue
            option = Variables.Securities[values]
            dataset = dataset.rename({"strike": str(option)})
            dataset["strike"] = dataset[str(option)]
            yield option, dataset



