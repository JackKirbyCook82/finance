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
from itertools import product
from functools import reduce
from collections import namedtuple as ntuple

from finance.variables import Variables, Querys
from support.meta import RegistryMeta
from support.calculations import Calculation, Equation, Variable
from support.mixins import Function, Emptying, Sizing, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StrategyCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class StrategySource(ntuple("Source", "axis position")): pass
class StrategyEquation(Equation, ABC, metaclass=RegistryMeta):
    t = Variable("current", np.datetime64, function=lambda tα, tβ: np.minimum(np.datetime64(tα, "ns"), np.datetime64(tβ, "ns")))
    q = Variable("size", np.float32, function=lambda qα, qβ: np.minimum(qα, qβ))
    x = Variable("underlying", np.float32, function=lambda xα, xβ: (xα + xβ) / 2)
    w = Variable("spot", np.float32, function=lambda y, ε: y * 100 - ε)
    wh = Variable("maximum", np.float32, function=lambda yh, ε: yh * 100 - ε)
    wl = Variable("minimum", np.float32, function=lambda yl, ε: yl * 100 - ε)

    tα = Variable("current", np.datetime64, locator=StrategySource("current", Variables.Positions.LONG))
    qα = Variable("size", np.float32, locator=StrategySource("size", Variables.Positions.LONG))
    xα = Variable("underlying", np.float32, locator=StrategySource("underlying", Variables.Positions.LONG))
    yα = Variable("price", np.float32, locator=StrategySource("price", Variables.Positions.LONG))
    kα = Variable("strike", np.float32, locator=StrategySource("strike", Variables.Positions.LONG))
    tβ = Variable("current", np.datetime64, locator=StrategySource("current", Variables.Positions.SHORT))
    qβ = Variable("size", np.float32, locator=StrategySource("size", Variables.Positions.SHORT))
    xβ = Variable("underlying", np.float32, locator=StrategySource("underlying", Variables.Positions.SHORT))
    yβ = Variable("price", np.float32, locator=StrategySource("price", Variables.Positions.SHORT))
    kβ = Variable("strike", np.float32, locator=StrategySource("strike", Variables.Positions.SHORT))

class VerticalEquation(StrategyEquation):
    y = Variable("spot", np.float32, function=lambda yα, yβ: - yα + yβ)

class CollarEquation(StrategyEquation):
    y = Variable("spot", np.float32, function=lambda yα, yβ, x: - yα + yβ - x)

class VerticalPutEquation(VerticalEquation, register=Variables.Strategies.Vertical.Put):
    yh = Variable("maximum", np.float32, function=lambda kα, kβ: np.maximum(kα - kβ, 0))
    yl = Variable("minimum", np.float32, function=lambda kα, kβ: np.minimum(kα - kβ, 0))

class VerticalCallEquation(VerticalEquation, register=Variables.Strategies.Vertical.Call):
    yh = Variable("maximum", np.float32, function=lambda kα, kβ: np.maximum(-kα + kβ, 0))
    yl = Variable("minimum", np.float32, function=lambda kα, kβ: np.minimum(-kα + kβ, 0))

class CollarLongEquation(CollarEquation, register=Variables.Strategies.Collar.Long):
    yh = Variable("maximum", np.float32, function=lambda kα, kβ: np.maximum(kα, kβ))
    yl = Variable("minimum", np.float32, function=lambda kα, kβ: np.minimum(kα, kβ))

class CollarShortEquation(CollarEquation, register=Variables.Strategies.Collar.Short):
    yh = Variable("maximum", np.float32, function=lambda kα, kβ: np.maximum(-kα, -kβ))
    yl = Variable("minimum", np.float32, function=lambda kα, kβ: np.minimum(-kα, -kβ))


class StrategyCalculation(Calculation, metaclass=RegistryMeta):
    def execute(self, options, *args, fees, **kwargs):
        positions = [option.position for option in options.keys()]
        assert len(set(positions)) == len(list(positions))
        options = {option.position: dataset for option, dataset in options.items()}
        variables, positions = ("price", "underlying", "strike", "size", "current"), list(Variables.Positions)
        sources = {StrategySource(variable, positions): options[positions][variable] for variable, position in product(variables, positions)}
        with self.equation(sources, fees=fees) as equation:
            yield equation.t()
            yield equation.q()
            yield equation.x()
            yield equation.y()
            yield equation.yl()
            yield equation.yh()


class VerticalCalculation(StrategyCalculation): pass
class CollarCalculation(StrategyCalculation): pass
class VerticalPutCalculation(VerticalCalculation, equation=VerticalPutEquation, register=Variables.Strategies.Vertical.Put): pass
class VerticalCallCalculation(VerticalCalculation, equation=VerticalCallEquation, register=Variables.Strategies.Vertical.Call): pass
class CollarLongCalculation(CollarCalculation, equation=CollarLongEquation, register=Variables.Strategies.Collar.LongV): pass
class CollarShortCalculation(CollarCalculation, equation=CollarShortEquation, register=Variables.Strategies.Collar.Shor): pass


class StrategyCalculator(Function, Logging, Sizing, Emptying):
    def __init__(self, *args, strategies=[], **kwargs):
        assert all([strategy in list(Variables.Strategies) for strategy in strategies])
        Function.__init__(self, *args, **kwargs)
        Logging.__init__(self, *args, **kwargs)
        strategies = list(dict(StrategyCalculation).keys()) if not bool(strategies) else list(strategies)
        calculations = dict(StrategyCalculation).items()
        calculations = {strategy: calculation(*args, **kwargs) for strategy, calculation in calculations if strategy in strategies}
        self.__calculations = calculations

    def execute(self, source, *args, **kwargs):
        assert isinstance(source, tuple)
        contract, options = source
        assert isinstance(contract, Querys.Contract) and isinstance(options, pd.DataFrame)
        if self.empty(options): return
        options = dict(self.options(options))
        for strategy, strategies in self.calculate(options, *args, **kwargs):
            size = self.size(strategies["size"])
            string = f"Calculated: {repr(self)}|{str(contract)}|{str(strategy)}[{size:.0f}]"
            self.logger.info(string)
            if self.empty(strategies["size"]): continue
            yield strategies

    def options(self, options):
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



