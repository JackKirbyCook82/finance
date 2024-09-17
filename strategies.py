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
from support.meta import ParametersMeta
from support.mixins import Sizing

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


class StrategyCalculation(Calculation, ABC, fields=["strategy"]):
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

class VerticalPutCalculation(StrategyCalculation, strategy=Variables.Strategies.Vertical.Put, equation=VerticalPutEquation): pass
class VerticalCallCalculation(StrategyCalculation, strategy=Variables.Strategies.Vertical.Call, equation=VerticalCallEquation): pass
class CollarLongCalculation(StrategyCalculation, strategy=Variables.Strategies.Collar.Long, equation=CollarLongEquation): pass
class CollarShortCalculation(StrategyCalculation, strategy=Variables.Strategies.Collar.Short, equation=CollarShortEquation): pass


class StrategyAxes(object, metaclass=ParametersMeta):
    security = ["instrument", "option", "position"]
    contract = ["ticker", "expire"]


class StrategyCalculator(Sizing):
    def __init__(self, *args, strategies=[], **kwargs):
        super().__init__(*args, **kwargs)
        calculations = {variables["strategy"]: calculation for variables, calculation in ODict(list(StrategyCalculation)).items()}
        strategies = calculations.keys() if not bool(strategies) else strategies
        calculations = {strategy: calculation for strategy, calculation in calculations.items() if strategy in strategies}
        self.__calculations = {strategy: calculation(*args, **kwargs) for strategy, calculation in calculations.items()}
        self.__variables = lambda **mapping: {key: xr.Variable(key, [value]).squeeze(key) for key, value in mapping.items()}
        self.__axes = StrategyAxes()
        self.__logger = __logger__

    def calculate(self, contract, options, *args, **kwargs):
        assert isinstance(contract, Contract) and isinstance(options, pd.DataFrame)
        options = ODict(list(self.options(options, *args, **kwargs)))
        strategies = ODict(list(self.strategies(options, *args, **kwargs)))
        strategies = list(strategies.values())
        assert all([isinstance(dataset, xr.Dataset) for dataset in strategies])
        size = self.size([dataset["size"] for dataset in strategies.values()])
        string = f"Calculated: {repr(self)}|{str(contract)}[{size:.0f}]"
        self.logger.info(string)
        return strategies

    def strategies(self, options, *args, **kwargs):
        if bool(options.empty): return
        for strategy, calculation in self.calculations.items():
            if not all([option in options.keys() for option in list(strategy.options)]): continue
            variables = self.variables(strategy=strategy)
            datasets = {option: options[option] for option in list(strategy.options)}
            strategies = calculation(datasets, *args, **kwargs)
            assert isinstance(strategies, xr.Dataset)
            if self.empty(strategies["size"]): continue
            strategies = strategies.assign_coords(variables)
            yield strategy, strategies

    def options(self, options, *args, **kwargs):
        if bool(options.empty): return
        index = self.axes.contract + self.axes.security + ["strike"]
        options = options.set_index(index, drop=True, inplace=False)
        options = xr.Dataset.from_dataframe(options)
        options = reduce(lambda content, axis: content.squeeze(axis), self.axes.contract, options)
        for values in product(*[options[axis].values for axis in self.axes.security]):
            dataset = options.sel(indexers={key: value for key, value in zip(self.axes.security, values)})
            dataset = dataset.drop_vars(self.axes.security, errors="ignore")
            if self.empty(dataset["size"]): continue
            option = Variables.Securities[values]
            dataset = dataset.rename({"strike": str(option)})
            dataset["strike"] = dataset[str(option)]
            yield option, dataset

    @property
    def calculations(self): return self.__calculations
    @property
    def variables(self): return self.__variables
    @property
    def logger(self): return self.__logger
    @property
    def axes(self): return self.__axes


