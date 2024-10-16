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

from finance.variables import Variables, Querys
from support.mixins import Emptying, Sizing, Logging, Pipelining, Sourcing
from support.calculations import Variable, Equation, Calculation
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


class StrategyCalculation(Calculation, ABC, metaclass=RegistryMeta):
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

class VerticalPutCalculation(StrategyCalculation, equation=VerticalPutEquation, register=Variables.Strategies.Vertical.Put): pass
class VerticalCallCalculation(StrategyCalculation, equation=VerticalCallEquation, register=Variables.Strategies.Vertical.Call): pass
class CollarLongCalculation(StrategyCalculation, equation=CollarLongEquation, register=Variables.Strategies.Collar.Long): pass
class CollarShortCalculation(StrategyCalculation, equation=CollarShortEquation, register=Variables.Strategies.Collar.Short): pass


class StrategyCalculator(Pipelining, Sourcing, Logging, Sizing, Emptying):
    def __init__(self, *args, strategies=[], **kwargs):
        assert all([strategy in list(Variables.Strategies) for strategy in strategies])
        Pipelining.__init__(self, *args, **kwargs)
        Logging.__init__(self, *args, **kwargs)
        strategies = list(dict(StrategyCalculation).keys()) if not bool(strategies) else list(strategies)
        calculations = dict(StrategyCalculation).items()
        calculations = {strategy: calculation(*args, **kwargs) for strategy, calculation in calculations if strategy in strategies}
        self.__calculations = calculations

    def execute(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if self.empty(options): return
        for contract, dataframe in self.source(options, keys=list(Querys.Contract)):
            contract = Querys.Contract(contract)
            if self.empty(dataframe): continue
            datasets = dict(self.options(dataframe, *args, **kwargs))
            for strategy, strategies in self.calculate(datasets, *args, **kwargs):
                size = self.size(strategies["size"])
                string = f"Calculated: {repr(self)}|{str(contract)}|{str(strategy)}[{size:.0f}]"
                self.logger.info(string)
                if self.empty(strategies["size"]): continue
                yield strategies

    def options(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        for security, dataframe in options.groupby(list(Variables.Security), sort=False):
            if self.empty(dataframe): continue
            security = Variables.Securities[security]
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
            yield strategy, strategies

    @property
    def calculations(self): return self.__calculations



