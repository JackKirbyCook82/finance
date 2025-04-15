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
from scipy.stats import norm
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
    whτ = Variable.Dependent("whτ", "maximum", np.float32, function=lambda yhτ, *, ε: yhτ * 100 - ε)
    weτ = Variable.Dependent("weτ", "expected", np.float32, function=lambda yeτ, * ε: yeτ * 100 - ε)
    wlτ = Variable.Dependent("wlτ", "minimum", np.float32, function=lambda ylτ, *, ε: ylτ * 100 - ε)
    wo = Variable.Dependent("wo", "spot", np.float32, function=lambda yo, *, ε: yo * 100 - ε)
    ε = Variable.Constant("ε", "fees", np.float32, locator="fees")

    σxo = Variable.Dependent("σxo", "volatility", np.float32, function=lambda σxα, σxβ: (σxα + σxβ) / 2)
    μxo = Variable.Dependent("μxo", "trend", np.float32, function=lambda μxα, μxβ: (μxα + μxβ) / 2)
    qxo = Variable.Dependent("qxo", "size", np.float32, function=lambda qxα, qxβ: (qxα + qxβ) / 2)
    yxo = Variable.Dependent("yxo", "underlying", np.float32, function=lambda yxα, yxβ: (yxα + yxβ) / 2)
    yxτ = Variable.Dependent("yxτ", "projection", np.float32, function=lambda yxo, μxo, τ: yxo + μxo * τ)

    kpα = Variable.Independent("kpα", "strike", np.float32, locator=StrategyLocator("strike", Securities.Options.Puts.Long))
    kpβ = Variable.Independent("kpβ", "strike", np.float32, locator=StrategyLocator("strike", Securities.Options.Puts.Short))
    kcα = Variable.Independent("kcα", "strike", np.float32, locator=StrategyLocator("strike", Securities.Options.Calls.Long))
    kcβ = Variable.Independent("kcβ", "strike", np.float32, locator=StrategyLocator("strike", Securities.Options.Calls.Short))

    ypα = Variable.Independent("ypα", "price", np.float32, locator=StrategyLocator("price", Securities.Options.Puts.Long))
    ypβ = Variable.Independent("ypβ", "price", np.float32, locator=StrategyLocator("price", Securities.Options.Puts.Short))
    ycα = Variable.Independent("ycα", "price", np.float32, locator=StrategyLocator("price", Securities.Options.Calls.Long))
    ycβ = Variable.Independent("ycβ", "price", np.float32, locator=StrategyLocator("price", Securities.Options.Calls.Short))
    yxα = Variable.Independent("yxα", "price", np.float32, locator=StrategyLocator("price", Securities.Stocks.Long))
    yxβ = Variable.Independent("yxβ", "price", np.float32, locator=StrategyLocator("price", Securities.Stocks.Short))

    qpα = Variable.Independent("qpα", "size", np.int32, locator=StrategyLocator("size", Securities.Options.Puts.Long))
    qpβ = Variable.Independent("qpβ", "size", np.int32, locator=StrategyLocator("size", Securities.Options.Puts.Short))
    qcα = Variable.Independent("qcα", "size", np.int32, locator=StrategyLocator("size", Securities.Options.Calls.Long))
    qcβ = Variable.Independent("qcβ", "size", np.int32, locator=StrategyLocator("size", Securities.Options.Calls.Short))
    qxα = Variable.Independent("qxα", "size", np.int32, locator=StrategyLocator("size", Securities.Stocks.Long))
    qxβ = Variable.Independent("qxβ", "size", np.int32, locator=StrategyLocator("size", Securities.Stocks.Short))

    σxα = Variable.Independent("σxα", "volatility", np.float32, locator=StrategyLocator("volatility", Securities.Stocks.Long))
    σxβ = Variable.Independent("σxβ", "volatility", np.float32, locator=StrategyLocator("volatility", Securities.Stocks.Short))
    μxα = Variable.Independent("μxα", "trend", np.float32, locator=StrategyLocator("trend", Securities.Stocks.Long))
    μxβ = Variable.Independent("μxβ", "trend", np.float32, locator=StrategyLocator("trend", Securities.Stocks.Short))

class VerticalPutEquation(StrategyEquation):
    yhτ = Variable.Dependent("yhτ", "maximum", np.float32, function=lambda kpα, kpβ: np.maximum(kpα - kpβ, 0))
    ylτ = Variable.Dependent("ylτ", "minimum", np.float32, function=lambda kpα, kpβ: np.minimum(kpα - kpβ, 0))
    yeτ = Variable.Dependent("yeτ", "expected", np.float32, function=lambda fpα, fcβ: fpα - fcβ)

    qo = Variable.Dependent("qo", "size", np.int32, function=lambda qpα, qpβ: np.minimum(qpα, qpβ))
    yo = Variable.Dependent("yo", "spot", np.float32, function=lambda ypα, ypβ: ypβ - ypα)
    ro = Variable.Dependent("ro", "revenue", np.float32, function=lambda ypβ: ypβ)
    eo = Variable.Dependent("eo", "expense", np.float32, function=lambda ypα: ypα)
    rτ = Variable.Dependent("rτ", "purchase", np.float32, function=lambda yxα: 0)
    eτ = Variable.Dependent("eτ", "borrow", np.float32, function=lambda yxβ: 0)

    zpα = Variable.Dependent("zpα", "zscore", np.float32, function=lambda kpα, yxτ, σxo: (kpα - yxτ) / σxo)
    zpβ = Variable.Dependent("zpβ", "zscore", np.float32, function=lambda kpβ, yxτ, σxo: (kpβ - yxτ) / σxo)
    Φpα = Variable.Dependent("Φpα", "normcdf", np.float32, function=lambda zpα: norm.cdf(zpα))
    Φpβ = Variable.Dependent("Φpβ", "normcdf", np.float32, function=lambda zpβ: norm.cdf(zpβ))
    φpα = Variable.Dependent("φpα", "normpdf", np.float32, function=lambda zpα: norm.pdf(zpα))
    φpβ = Variable.Dependent("φpβ", "normpdf", np.float32, function=lambda zpβ: norm.pdf(zpβ))
    fpα = Variable.Dependent("fpα", "function", np.float32, function=lambda kpα, Φpα, φpα, yxτ, σxo: (kpα - yxτ) * Φpα + σxo * φpα)
    fcβ = Variable.Dependent("fcβ", "function", np.float32, function=lambda kpβ, Φpβ, φpβ, yxτ, σxo: (kpβ - yxτ) * Φpβ + σxo * φpβ)

class VerticalCallEquation(StrategyEquation):
    yhτ = Variable.Dependent("yhτ", "maximum", np.float32, function=lambda kcα, kcβ: np.maximum(kcβ - kcα, 0))
    ylτ = Variable.Dependent("ylτ", "minimum", np.float32, function=lambda kcα, kcβ: np.minimum(kcβ - kcα, 0))
    yeτ = Variable.Dependent("yeτ", "expected", np.float32, function=lambda fpα, fcβ: fpα - fcβ)

    qo = Variable.Dependent("qo", "size", np.int32, function=lambda qcα, qcβ: np.minimum(qcα, qcβ))
    yo = Variable.Dependent("yo", "spot", np.float32, function=lambda ycα, ycβ: ycβ - ycα)
    ro = Variable.Dependent("ro", "revenue", np.float32, function=lambda ycβ: ycβ)
    eo = Variable.Dependent("eo", "expense", np.float32, function=lambda ycα: ycα)
    rτ = Variable.Dependent("rτ", "purchase", np.float32, function=lambda yxα: 0)
    eτ = Variable.Dependent("eτ", "borrow", np.float32, function=lambda yxβ: 0)

    zcα = Variable.Dependent("zcα", "zscore", np.float32, function=lambda kcα, yxτ, σxo: (kcα - yxτ) / σxo)
    zcβ = Variable.Dependent("zcβ", "zscore", np.float32, function=lambda kcβ, yxτ, σxo: (kcβ - yxτ) / σxo)
    Φcα = Variable.Dependent("Φcα", "normcdf", np.float32, function=lambda zcα: norm.cdf(zcα))
    Φcβ = Variable.Dependent("Φcβ", "normcdf", np.float32, function=lambda zcβ: norm.cdf(zcβ))
    φcα = Variable.Dependent("φcα", "normpdf", np.float32, function=lambda zcα: norm.pdf(zcα))
    φcβ = Variable.Dependent("φcβ", "normpdf", np.float32, function=lambda zcβ: norm.pdf(zcβ))
    fpα = Variable.Dependent("fpα", "function", np.float32, function=lambda kcα, Φcα, φcα, yxτ, σxo: (yxτ - kcα) * Φcα + σxo * φcα)
    fcβ = Variable.Dependent("fcβ", "function", np.float32, function=lambda kcβ, Φcβ, φcβ, yxτ, σxo: (yxτ - kcβ) * Φcβ + σxo * φcβ)

class CollarLongEquation(StrategyEquation):
    yhτ = Variable.Dependent("yhτ", "maximum", np.float32, function=lambda kpα, kcβ: + np.maximum(kpα, kcβ))
    ylτ = Variable.Dependent("ylτ", "minimum", np.float32, function=lambda kpα, kcβ: + np.minimum(kpα, kcβ))
    yeτ = Variable.Dependent("yeτ", "expected", np.float32, function=lambda yxτ, fpα, fcβ: fpα - fcβ + yxτ)

    qo = Variable.Dependent("qo", "size", np.int32, function=lambda qpα, qcβ: np.minimum(qpα, qcβ))
    yo = Variable.Dependent("yo", "spot", np.float32, function=lambda ypα, ycβ, yxα: ycβ - ypα - yxα)
    ro = Variable.Dependent("ro", "revenue", np.float32, function=lambda ycβ: ycβ)
    eo = Variable.Dependent("eo", "expense", np.float32, function=lambda ypα: ypα)
    rτ = Variable.Dependent("rτ", "purchase", np.float32, function=lambda yxα: yxα)
    eτ = Variable.Dependent("eτ", "borrow", np.float32, function=lambda yxβ: 0)

    zpα = Variable.Dependent("zpα", "zscore", np.float32, function=lambda kpα, yxτ, σxo: (kpα - yxτ) / σxo)
    zcβ = Variable.Dependent("zcβ", "zscore", np.float32, function=lambda kcβ, yxτ, σxo: (kcβ - yxτ) / σxo)
    Φpα = Variable.Dependent("Φpα", "normcdf", np.float32, function=lambda zpα: norm.cdf(zpα))
    Φcβ = Variable.Dependent("Φcβ", "normcdf", np.float32, function=lambda zcβ: norm.cdf(zcβ))
    φpα = Variable.Dependent("φpα", "normpdf", np.float32, function=lambda zpα: norm.pdf(zpα))
    φcβ = Variable.Dependent("φcβ", "normpdf", np.float32, function=lambda zcβ: norm.pdf(zcβ))
    fpα = Variable.Dependent("fpα", "function", np.float32, function=lambda kpα, Φpα, φpα, yxτ, σxo: (kpα - yxτ) * Φpα + σxo * φpα)
    fcβ = Variable.Dependent("fcβ", "function", np.float32, function=lambda kcβ, Φcβ, φcβ, yxτ, σxo: (yxτ - kcβ) * Φcβ + σxo * φcβ)

class CollarShortEquation(StrategyEquation):
    yhτ = Variable.Dependent("yhτ", "maximum", np.float32, function=lambda kcα, kpβ: - np.minimum(kcα, kpβ))
    ylτ = Variable.Dependent("ylτ", "minimum", np.float32, function=lambda kcα, kpβ: - np.maximum(kcα, kpβ))
    yeτ = Variable.Dependent("yeτ", "expected", np.float32, function=lambda yxτ, fcα, fpβ: fcα - fpβ - yxτ)

    qo = Variable.Dependent("qo", "size", np.int32, function=lambda qcα, qpβ: np.minimum(qcα, qpβ))
    yo = Variable.Dependent("yo", "spot", np.float32, function=lambda ycα, ypβ, yxβ: ypβ - ycα + yxβ)
    ro = Variable.Dependent("ro", "revenue", np.float32, function=lambda ypβ: ypβ)
    eo = Variable.Dependent("eo", "expense", np.float32, function=lambda ycα: ycα)
    rτ = Variable.Dependent("rτ", "purchase", np.float32, function=lambda yxα: 0)
    eτ = Variable.Dependent("eτ", "borrow", np.float32, function=lambda yxβ: yxβ)

    zcα = Variable.Dependent("zcα", "zscore", np.float32, function=lambda kcα, yxτ, σxo: (kcα - yxτ) / σxo)
    zpβ = Variable.Dependent("zpβ", "zscore", np.float32, function=lambda kpβ, yxτ, σxo: (kpβ - yxτ) / σxo)
    Φcα = Variable.Dependent("Φcα", "normcdf", np.float32, function=lambda zcα: norm.cdf(zcα))
    Φpβ = Variable.Dependent("Φpβ", "normcdf", np.float32, function=lambda zpβ: norm.cdf(zpβ))
    φcα = Variable.Dependent("φcα", "normpdf", np.float32, function=lambda zcα: norm.pdf(zcα))
    φpβ = Variable.Dependent("φpβ", "normpdf", np.float32, function=lambda zpβ: norm.pdf(zpβ))
    fcα = Variable.Dependent("fcα", "function", np.float32, function=lambda kcα, Φcα, φcα, yxτ, σxo: (yxτ - kcα) * Φcα + σxo * φcα)
    fpβ = Variable.Dependent("fpβ", "function", np.float32, function=lambda kpβ, Φpβ, φpβ, yxτ, σxo: (kpβ - yxτ) * Φpβ + σxo * φpβ)

class StrategyCalculation(Calculation, ABC, metaclass=RegistryMeta):
    def __init_subclass__(cls, *args, strategy, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls.__strategy__ = strategy

    def execute(self, stocks, options, *args, fees, **kwargs):
        assert all([stock in stocks.keys() for stock in self.strategy.stocks])
        assert all([option in options.keys() for option in self.strategy.options])
        stocks = {StrategyLocator(axis, security): dataset[axis] for security, dataset in stocks.items() for axis in ("price", "size", "trend", "volatility")}
        options = {StrategyLocator(axis, security): dataset[axis] for security, dataset in options.items() for axis in ("price", "strike", "size")}
        with self.equation(stocks | options, fees=fees) as equation:
            yield equation.yxo()
            yield equation.wlτ()
            yield equation.weτ()
            yield equation.whτ()
            yield equation.wo()
            yield equation.qo()
            yield equation.ro()
            yield equation.eo()
            yield equation.rτ()
            yield equation.eτ()

    @property
    def strategy(self): return type(self).__strategy__

class VerticalPutCalculation(StrategyCalculation, equation=VerticalPutEquation, strategy=Strategies.Verticals.Put, register=Strategies.Verticals.Put): pass
class VerticalCallCalculation(StrategyCalculation, equation=VerticalCallEquation, strategy=Strategies.Verticals.Call, register=Strategies.Verticals.Call): pass
class CollarLongCalculation(StrategyCalculation, equation=CollarLongEquation, strategy=Strategies.Collars.Long, register=Strategies.Collars.Long): pass
class CollarShortCalculation(StrategyCalculation, equation=CollarShortEquation, strategy=Strategies.Collars.Short, register=Strategies.Collars.Short): pass


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
        for settlement, secondary in self.partition(options, by=Querys.Settlement):
            primary = stocks.where(stocks["ticker"] == settlement.ticker).dropna(how="all", inplace=False)
            primary = dict(self.stocks(primary, *args, **kwargs))
            secondary = dict(self.options(secondary, *args, **kwargs))
            strategies = self.calculate(primary, secondary, *args, **kwargs)
            for strategy, dataset in strategies.items():
                size = self.size(dataset, "size")
                self.console(f"{str(settlement)}|{str(strategy)}[{int(size):.0f}]")
                if self.empty(dataset, "size"): continue
                yield dataset

    def calculate(self, stocks, options, *args, **kwargs):
        strategies = dict(self.calculator(stocks, options, *args, **kwargs))
        return strategies

    def calculator(self, stocks, options, *args, **kwargs):
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
        stocks = stocks.set_index(list(Querys.Symbol) + list(Variables.Securities.Security), drop=True, inplace=False)
        stocks = xr.Dataset.from_dataframe(stocks)
        columns = set(Querys.Symbol) | set(Variables.Securities.Security) - {"position"}
        stocks = reduce(lambda content, axis: content.squeeze(axis), list(columns), stocks)
        for position in list(Variables.Securities.Position):
            security = Securities([Variables.Securities.Instrument.STOCK, Variables.Securities.Option.EMPTY, position])
            dataset = stocks.sel(position=position).drop_vars(list(Variables.Securities.Security))
            yield security, dataset

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



