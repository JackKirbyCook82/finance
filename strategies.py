# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Strategy Objects
@author: Jack Kirby Cook

"""

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


class StrategyLocator(ntuple("Locator", "axis security")): pass
class StrategyEquation(Equation, ABC, datatype=xr.DataArray, vectorize=True):
    x = Variable.Dependent("x", "underlying", np.float32, function=lambda xα, xβ: np.divide(xα + xβ, 2))
    w = Variable.Dependent("w", "spot", np.float32, function=lambda y, *, ε: y * 100 - ε)
    wh = Variable.Dependent("wh", "maximum", np.float32, function=lambda yh, *, ε: yh * 100 - ε)
    wl = Variable.Dependent("wl", "minimum", np.float32, function=lambda yl, *, ε: yl * 100 - ε)

    ypα = Variable.Independent("ypα", "price", np.float32, locator=StrategyLocator("price", Securities.Options.Puts.Long))
    qpα = Variable.Independent("qpα", "size", np.float32, locator=StrategyLocator("size", Securities.Options.Puts.Long))
    kpα = Variable.Independent("kpα", "strike", np.float32, locator=StrategyLocator("strike", Securities.Options.Puts.Long))
    qpα = Variable.Independent("qpα", "size", np.float32, locator=StrategyLocator("size", Securities.Options.Puts.Long))

    ypβ = Variable.Independent("ypβ", "price", np.float32, locator=StrategyLocator("price", Securities.Options.Puts.Short))
    qpβ = Variable.Independent("qpβ", "size", np.float32, locator=StrategyLocator("size", Securities.Options.Puts.Short))
    kpβ = Variable.Independent("kpβ", "strike", np.float32, locator=StrategyLocator("strike", Securities.Options.Puts.Short))
    qpβ = Variable.Independent("qpβ", "size", np.float32, locator=StrategyLocator("size", Securities.Options.Puts.Short))

    ycα = Variable.Independent("ycα", "price", np.float32, locator=StrategyLocator("price", Securities.Options.Calls.Long))
    qcα = Variable.Independent("qcα", "size", np.float32, locator=StrategyLocator("size", Securities.Options.Calls.Long))
    kcα = Variable.Independent("kcα", "strike", np.float32, locator=StrategyLocator("strike", Securities.Options.Calls.Long))
    qcα = Variable.Independent("qcα", "size", np.float32, locator=StrategyLocator("size", Securities.Options.Calls.Long))

    ycβ = Variable.Independent("ycβ", "price", np.float32, locator=StrategyLocator("price", Securities.Options.Calls.Short))
    qcβ = Variable.Independent("qcβ", "size", np.float32, locator=StrategyLocator("size", Securities.Options.Calls.Short))
    kcβ = Variable.Independent("kcβ", "strike", np.float32, locator=StrategyLocator("strike", Securities.Options.Calls.Short))
    qcβ = Variable.Independent("qcβ", "size", np.float32, locator=StrategyLocator("size", Securities.Options.Calls.Short))

    xα = Variable.Independent("xα", "underlying", np.float32, locator=Securities.Stocks.Long)
    xβ = Variable.Independent("xβ", "underlying", np.float32, locator=Securities.Stocks.Short)
    ε = Variable.Constant("ε", "fees", np.float32, locator="fees")

class VerticalPutEquation(StrategyEquation):
    q = Variable.Dependent("q", "size", np.float32, function=lambda qpα, qpβ: np.minimum(qpα, qpβ))
    y = Variable.Dependent("y", "spot", np.float32, function=lambda ypα, ypβ: ypβ - ypα)
    yh = Variable.Dependent("yh", "maximum", np.float32, function=lambda kpα, kpβ: np.maximum(kpα - kpβ, 0))
    yl = Variable.Dependent("yl", "minimum", np.float32, function=lambda kpα, kpβ: np.minimum(kpα - kpβ, 0))

class VerticalCallEquation(StrategyEquation):
    q = Variable.Dependent("q", "size", np.float32, function=lambda qcα, qcβ: np.minimum(qcα, qcβ))
    y = Variable.Dependent("y", "spot", np.float32, function=lambda ycα, ycβ: ycβ - ycα)
    yh = Variable.Dependent("yh", "maximum", np.float32, function=lambda kcα, kcβ: np.maximum(kcβ - kcα, 0))
    yl = Variable.Dependent("yl", "minimum", np.float32, function=lambda kcα, kcβ: np.minimum(kcβ - kcα, 0))

class CollarLongEquation(StrategyEquation):
    q = Variable.Dependent("q", "size", np.float32, function=lambda qcα, qpβ: np.minimum(qcα, qpβ))
    y = Variable.Dependent("y", "spot", np.float32, function=lambda xα, ycα, ypβ: ypβ - ycα - xα)
    yh = Variable.Dependent("yh", "maximum", np.float32, function=lambda kcα, kpβ: + np.maximum(kcα, kpβ))
    yl = Variable.Dependent("yl", "minimum", np.float32, function=lambda kcα, kpβ: + np.minimum(kcα, kpβ))

class CollarShortEquation(StrategyEquation):
    q = Variable.Dependent("q", "size", np.float32, function=lambda kpα, kcβ: np.minimum(kpα, kcβ))
    y = Variable.Dependent("y", "spot", np.float32, function=lambda xβ, ypα, ycβ: ycβ - ypα + xβ)
    yh = Variable.Dependent("yh", "maximum", np.float32, function=lambda kpα, kcβ: - np.maximum(kpα, kcβ))
    yl = Variable.Dependent("yl", "minimum", np.float32, function=lambda kpα, kcβ: - np.minimum(kpα, kcβ))


class StrategyCalculation(Calculation, ABC, metaclass=RegistryMeta):
    def execute(self, stocks, options, *args, fees, **kwargs):
        options = {StrategyLocator(axis, security): dataset[axis] for security, dataset in options.items() for axis in ("price", "strike", "size")}
        with self.equation(stocks | options, fees=fees) as equation:
            yield equation.q()
            yield equation.w()
            yield equation.wl()
            yield equation.wh()
            yield equation.x()

class VerticalPutCalculation(StrategyCalculation, equation=VerticalPutEquation, register=Strategies.Verticals.Put): pass
class VerticalCallCalculation(StrategyCalculation, equation=VerticalCallEquation, register=Strategies.Verticals.Call): pass
class CollarLongCalculation(StrategyCalculation, equation=CollarLongEquation, register=Strategies.Collars.Long): pass
class CollarShortCalculation(StrategyCalculation, equation=CollarShortEquation, register=Strategies.Collars.Short): pass


class StrategyCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, strategies=[], **kwargs):
        assert all([strategy in list(Strategies) for strategy in list(strategies)])
        super().__init__(*args, **kwargs)
        strategies = list(dict(StrategyCalculation).keys()) if not bool(strategies) else list(strategies)
        calculations = {strategy: calculation(*args, **kwargs) for strategy, calculation in dict(StrategyCalculation).items() if strategy in strategies}
        self.__calculations = calculations

    def execute(self, stocks, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if self.empty(options): return
        for settlement, primary in self.partition(options, by=Querys.Settlement):
            secondary = stocks.where(stocks["ticker"] == settlement.ticker).dropna(how="all", inplace=False)
            primary = dict(self.options(primary, *args, **kwargs))
            secondary = dict(self.stocks(secondary, *args, **kwargs))
            strategies = self.calculate(primary, secondary, *args, **kwargs)
            for strategy, dataset in strategies.items():
                size = self.size(dataset, "size")
                self.console(f"{str(settlement)}|{str(strategy)}[{int(size):.0f}]")
                if self.empty(dataset, "size"): continue
                yield dataset

    def calculate(self, options, *args, **kwargs):
        strategies = dict(self.calculator(options, *args, **kwargs))
        return strategies

    def calculator(self, options, stocks, *args, **kwargs):
        for strategy, calculation in self.calculations.items():
            if not all([stock in stocks.keys() for stock in list(strategy.stocks)]): continue
            if not all([option in options.keys() for option in list(strategy.options)]): continue
            strategies = calculation(stocks, options, *args, **kwargs)
            assert isinstance(strategies, xr.Dataset)
            strategies = strategies.assign_coords({"strategy": xr.Variable("strategy", [strategy]).squeeze("strategy")})
            for field in list(Querys.Settlement): strategies = strategies.expand_dims(field)
            yield strategy, strategies

    @staticmethod
    def stocks(stocks, *args, **kwargs):
        for index, series in stocks.iterrows():
            security = (series.instrument, Variables.Securities.Option.EMPTY, series.position)
            security = Securities(security)
            value = np.float32(series.price)
            yield security, value

    @staticmethod
    def options(options, *args, **kwargs):
        for security, dataframe in options.groupby(list(Variables.Securities.Security), sort=False):
            if dataframe.empty: continue
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



