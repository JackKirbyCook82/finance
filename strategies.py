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

from finance.variables import Variables, Contract
from support.calculations import Variable, Equation, Calculation
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StrategyCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class StrategyVariables(object):
    axes = {Variables.Querys.CONTRACT: ["ticker", "expire"], Variables.Datasets.SECURITY: ["instrument", "option", "position"]}

    def __init__(self, *args, **kwargs):
        self.security = self.axes[Variables.Datasets.SECURITY]
        self.contract = self.axes[Variables.Querys.CONTRACT]
        self.index = self.security + self.contract + ["strike"]


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


class StrategyCalculator(object):
    def __init__(self, *args, strategies=[], **kwargs):
        strategies = list(dict(StrategyCalculation).keys()) if not bool(strategies) else list(strategies)
        calculations = dict(StrategyCalculation).items()
        self.__calculations = {strategy: calculation(*args, **kwargs) for strategy, calculation in calculations if strategy in strategies}
        self.__variables = StrategyVariables(*args, **kwargs)

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        for contract, dataframe in self.contracts(options):
            for strategy, strategies in self.execute(dataframe, *args, **kwargs):
                size = np.count_nonzero(~np.isnan(strategies["size"].values))
                string = f"Calculated: {repr(self)}|{str(contract)}|{str(strategy)}[{size:.0f}]"
                self.logger.info(string)
                if not bool(np.count_nonzero(~np.isnan(strategies["size"].values))): continue
                yield strategies

    def contracts(self, options):
        assert isinstance(options, pd.DataFrame)
        for (ticker, expire), dataframe in options.groupby(self.variables.contract):
            if bool(dataframe.empty): continue
            contract = Contract(ticker, expire)
            yield contract, dataframe

    def options(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        for (instrument, option, position), dataframe in options.groupby(self.variables.security):
            if bool(dataframe.empty): continue
            security = Variables.Securities[instrument, option, position]
            dataframe = dataframe.set_index(self.variables.index, drop=True, inplace=False)
            dataset = xr.Dataset.from_dataframe(dataframe)
            dataset = reduce(lambda content, axis: content.squeeze(axis), self.variables.contract, dataset)
            dataset = dataset.drop_vars(self.variables.security, errors="ignore")
            dataset = dataset.rename({"strike": str(security)})
            dataset["strike"] = dataset[str(security)]
            yield security, dataset

    def execute(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        options = dict(self.options(options, *args, **kwargs))
        for strategy, strategies in self.calculate(options, *args, **kwargs):
            yield strategy, strategies

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
            yield strategy, strategies

    @property
    def calculations(self): return self.__calculations
    @property
    def variables(self): return self.__variables



