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

from finance.variables import Querys, Variables, Strategies, Securities
from support.mixins import Emptying, Sizing, Partition, Logging
from support.calculations import Calculation, Equation, Variable
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StrategyCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class StrategyLocator(ntuple("Locator", "axis security")): pass
class StrategyCollection(ntuple("Collection", "payoff underlying greeks")): pass


class StrategyEquationMeta(RegistryMeta, type(Equation)): pass
class StrategyEquation(Equation, ABC, datatype=xr.DataArray, vectorize=False):
    ypα = Variable.Independent("ypα", "spot", np.float32, locator=StrategyLocator(Securities.Options.Puts.Long, "spot"))
    ypβ = Variable.Independent("ypβ", "spot", np.float32, locator=StrategyLocator(Securities.Options.Puts.Short, "spot"))
    ycα = Variable.Independent("ycα", "spot", np.float32, locator=StrategyLocator(Securities.Options.Calls.Long, "spot"))
    ycβ = Variable.Independent("ycβ", "spot", np.float32, locator=StrategyLocator(Securities.Options.Calls.Short, "spot"))

    qpα = Variable.Independent("qpα", "size", np.int32, locator=StrategyLocator(Securities.Options.Puts.Long, "size"))
    qpβ = Variable.Independent("qpβ", "size", np.int32, locator=StrategyLocator(Securities.Options.Puts.Short, "size"))
    qcα = Variable.Independent("qcα", "size", np.int32, locator=StrategyLocator(Securities.Options.Calls.Long, "size"))
    qcβ = Variable.Independent("qcβ", "size", np.int32, locator=StrategyLocator(Securities.Options.Calls.Short, "size"))

    def __init_subclass__(cls, *args, **kwargs):
        cls.strategy = kwargs.get("strategy", getattr(cls, "strategy", None))

    def execute(self, *args, **kwargs):
        yield from super().execute(*args, **kwargs)
        yield self.yo()
        yield self.qo()


class VerticalPutStrategyEquation(StrategyEquation, strategy=Strategies.Verticals.Put):
    yo = Variable.Dependent("yo", "spot", np.float32, function=lambda ypα, ypβ: ypβ + ypα)
    qo = Variable.Dependent("qo", "size", np.int32, function=lambda qpα, qpβ: np.minimum(qpα, qpβ))

class VerticalCallStrategyEquation(StrategyEquation, strategy=Strategies.Verticals.Call):
    yo = Variable.Dependent("yo", "spot", np.float32, function=lambda ycα, ycβ: ycβ + ycα)
    qo = Variable.Dependent("qo", "size", np.int32, function=lambda qcα, qcβ: np.minimum(qcα, qcβ))

class CollarLongStrategyEquation(StrategyEquation, strategy=Strategies.Collars.Long):
    yo = Variable.Dependent("yo", "spot", np.float32, function=lambda ypα, ycβ, xo: ycβ + ypα - xo)
    qo = Variable.Dependent("qo", "size", np.int32, function=lambda qpα, qcβ: np.minimum(qpα, qcβ))

class CollarShortStrategyEquation(StrategyEquation, strategy=Strategies.Collars.Short):
    yo = Variable.Dependent("yo", "spot", np.float32, function=lambda ycα, ypβ, xo: ypβ + ycα + xo)
    qo = Variable.Dependent("qo", "size", np.int32, function=lambda qcα, qpβ: np.minimum(qcα, qpβ))


class PayoffEquation(StrategyEquation, metaclass=StrategyEquationMeta):
    yk = Variable.Dependent("yk", "breakeven", np.float32, function=lambda yo, ymin, ymax: xr.where(np.negative(yo) <= ymax, xr.where(np.negative(yo) >= ymin, np.negative(yo), np.NaN), np.NaN))

    kpα = Variable.Independent("kpα", "strike", np.float32, locator=StrategyLocator(Securities.Options.Puts.Long, "strike"))
    kpβ = Variable.Independent("kpβ", "strike", np.float32, locator=StrategyLocator(Securities.Options.Puts.Short, "strike"))
    kcα = Variable.Independent("kcα", "strike", np.float32, locator=StrategyLocator(Securities.Options.Calls.Long, "strike"))
    kcβ = Variable.Independent("kcβ", "strike", np.float32, locator=StrategyLocator(Securities.Options.Calls.Short, "strike"))

    def execute(self, *args, **kwargs):
        yield from super().execute(*args, **kwargs)
        yield self.ymax()
        yield self.ymin()
        yield self.xk()
        yield self.kα()
        yield self.kβ()


class VerticalPutPayoffEquation(PayoffEquation, VerticalPutStrategyEquation, register=Strategies.Verticals.Put):
    xk = Variable.Dependent("xk", "breakeven", np.float32, function=lambda yk, xmax: xmax - np.abs(yk))
    mk = Variable.Dependent("mk", "breakeven", np.float32, function=lambda kpα, kpβ: np.sign(kpβ - kpα))
    ymax = Variable.Dependent("ymax", "maximum", np.float32, function=lambda kpα, kpβ: np.maximum(kpα - kpβ, 0))
    ymin = Variable.Dependent("ymin", "minimum", np.float32, function=lambda kpα, kpβ: np.minimum(kpα - kpβ, 0))
    xmax = Variable.Dependent("xmax", "strike", np.float32, function=lambda kpα, kpβ: np.maximum(kpα, kpβ))
    xmin = Variable.Dependent("xmin", "strike", np.float32, function=lambda kpα, kpβ: np.minimum(kpα, kpβ))
    kα = Variable.Dependent("kα", "long", np.float32, function=lambda kpα: kpα)
    kβ = Variable.Dependent("kβ", "short", np.float32, function=lambda kpβ: kpβ)

class VerticalCallPayoffEquation(PayoffEquation, VerticalCallStrategyEquation, register=Strategies.Verticals.Call):
    xk = Variable.Dependent("xk", "breakeven", np.float32, function=lambda yk, xmin: xmin + np.abs(yk))
    mk = Variable.Dependent("mk", "breakeven", np.float32, function=lambda kcα, kcβ: np.sign(kcβ - kcα))
    ymax = Variable.Dependent("ymax", "maximum", np.float32, function=lambda kcα, kcβ: np.maximum(kcβ - kcα, 0))
    ymin = Variable.Dependent("ymin", "minimum", np.float32, function=lambda kcα, kcβ: np.minimum(kcβ - kcα, 0))
    xmax = Variable.Dependent("xmax", "strike", np.float32, function=lambda kcα, kcβ: np.maximum(kcα, kcβ))
    xmin = Variable.Dependent("xmin", "strike", np.float32, function=lambda kcα, kcβ: np.minimum(kcα, kcβ))
    kα = Variable.Dependent("kα", "long", np.float32, function=lambda kcα: kcα)
    kβ = Variable.Dependent("kβ", "short", np.float32, function=lambda kcβ: kcβ)

class CollarLongPayoffEquation(PayoffEquation, CollarLongStrategyEquation, register=Strategies.Collars.Long):
    xk = Variable.Dependent("xk", "breakeven", np.float32, function=lambda yk, mk, xmin, ymin, ymax: xmin + (yk - ymin) * np.abs(mk + 1) / 2 + (ymax - yk) * np.abs(mk - 1) / 2)
    mk = Variable.Dependent("mk", "breakeven", np.float32, function=lambda kpα, kcβ: np.sign(kcβ - kpα))
    ymax = Variable.Dependent("ymax", "maximum", np.float32, function=lambda kpα, kcβ: + np.maximum(kpα, kcβ))
    ymin = Variable.Dependent("ymin", "minimum", np.float32, function=lambda kpα, kcβ: + np.minimum(kpα, kcβ))
    xmax = Variable.Dependent("xmax", "strike", np.float32, function=lambda kpα, kcβ: np.maximum(kpα, kcβ))
    xmin = Variable.Dependent("xmin", "strike", np.float32, function=lambda kpα, kcβ: np.minimum(kpα, kcβ))
    kα = Variable.Dependent("kα", "long", np.float32, function=lambda kpα: kpα)
    kβ = Variable.Dependent("kβ", "short", np.float32, function=lambda kcβ: kcβ)

class CollarShortPayoffEquation(PayoffEquation, CollarShortStrategyEquation, register=Strategies.Collars.Short):
    xk = Variable.Dependent("xk", "breakeven", np.float32, function=lambda yk, mk, xmin, ymin, ymax: xmin + (yk - ymin) * np.abs(mk + 1) / 2 + (ymax - yk) * np.abs(mk - 1) / 2)
    mk = Variable.Dependent("mk", "breakeven", np.float32, function=lambda kcα, kpβ: np.sign(kpβ - kcα))
    ymax = Variable.Dependent("ymax", "maximum", np.float32, function=lambda kcα, kpβ: - np.minimum(kcα, kpβ))
    ymin = Variable.Dependent("ymin", "minimum", np.float32, function=lambda kcα, kpβ: - np.maximum(kcα, kpβ))
    xmax = Variable.Dependent("xmax", "strike", np.float32, function=lambda kcα, kpβ: np.maximum(kcα, kpβ))
    xmin = Variable.Dependent("xmin", "strike", np.float32, function=lambda kcα, kpβ: np.minimum(kcα, kpβ))
    kα = Variable.Dependent("kα", "long", np.float32, function=lambda kcα: kcα)
    kβ = Variable.Dependent("kβ", "short", np.float32, function=lambda kpβ: kpβ)


class UnderlyingEquation(StrategyEquation, metaclass=StrategyEquationMeta):
    xpα = Variable.Independent("xpα", "underlying", np.float32, locator=StrategyLocator(Securities.Options.Puts.Long, "underlying"))
    xpβ = Variable.Independent("xpβ", "underlying", np.float32, locator=StrategyLocator(Securities.Options.Puts.Short, "underlying"))
    xcα = Variable.Independent("xcα", "underlying", np.float32, locator=StrategyLocator(Securities.Options.Calls.Long, "underlying"))
    xcβ = Variable.Independent("xcβ", "underlying", np.float32, locator=StrategyLocator(Securities.Options.Calls.Short, "underlying"))

    μpα = Variable.Independent("μpα", "trend", np.float32, locator=StrategyLocator(Securities.Options.Puts.Long, "trend"))
    μpβ = Variable.Independent("μpβ", "trend", np.float32, locator=StrategyLocator(Securities.Options.Puts.Short, "trend"))
    μcα = Variable.Independent("μcα", "trend", np.float32, locator=StrategyLocator(Securities.Options.Calls.Long, "trend"))
    μcβ = Variable.Independent("μcβ", "trend", np.float32, locator=StrategyLocator(Securities.Options.Calls.Short, "trend"))

    δpα = Variable.Independent("δpα", "volatility", np.float32, locator=StrategyLocator(Securities.Options.Puts.Long, "volatility"))
    δpβ = Variable.Independent("δpβ", "volatility", np.float32, locator=StrategyLocator(Securities.Options.Puts.Short, "volatility"))
    δcα = Variable.Independent("δcα", "volatility", np.float32, locator=StrategyLocator(Securities.Options.Calls.Long, "volatility"))
    δcβ = Variable.Independent("δcβ", "volatility", np.float32, locator=StrategyLocator(Securities.Options.Calls.Short, "volatility"))

    def execute(self, *args, **kwargs):
        yield from super().execute(*args, **kwargs)
        yield self.xo()
        yield self.μo()
        yield self.δo()


class VerticalPutUnderlyingEquation(UnderlyingEquation, VerticalPutStrategyEquation, register=Strategies.Verticals.Put):
    xo = Variable.Dependent("xo", "underlying", np.float32, function=lambda xpα, xpβ: np.divide(xpα + xpβ, 2))
    μo = Variable.Dependent("μo", "trend", np.float32, function=lambda μpα, μpβ: np.divide(μpα + μpβ, 2))
    δo = Variable.Dependent("δo", "volatility", np.float32, function=lambda δpα, δpβ: np.divide(δpα + δpβ, 2))

class VerticalCallUnderlyingEquation(UnderlyingEquation, VerticalCallStrategyEquation, register=Strategies.Verticals.Call):
    xo = Variable.Dependent("xo", "underlying", np.float32, function=lambda xcα, xcβ: np.divide(xcα + xcβ, 2))
    μo = Variable.Dependent("μo", "trend", np.float32, function=lambda μcα, μcβ: np.divide(μcα + μcβ, 2))
    δo = Variable.Dependent("δo", "volatility", np.float32, function=lambda δcα, δcβ: np.divide(δcα + δcβ, 2))

class CollarLongUnderlyingEquation(UnderlyingEquation, CollarLongStrategyEquation, register=Strategies.Collars.Long):
    xo = Variable.Dependent("xo", "underlying", np.float32, function=lambda xpα, xcβ: np.divide(xpα + xcβ, 2))
    μo = Variable.Dependent("μo", "trend", np.float32, function=lambda μpα, μcβ: np.divide(μpα + μcβ, 2))
    δo = Variable.Dependent("δo", "volatility", np.float32, function=lambda δpα, δcβ: np.divide(δpα + δcβ, 2))

class CollarShortUnderlyingEquation(UnderlyingEquation, CollarShortStrategyEquation, register=Strategies.Collars.Short):
    xo = Variable.Dependent("xo", "underlying", np.float32, function=lambda xcα, xpβ: np.divide(xcα + xpβ, 2))
    μo = Variable.Dependent("μo", "trend", np.float32, function=lambda μcα, μpβ: np.divide(μcα + μpβ, 2))
    δo = Variable.Dependent("δo", "volatility", np.float32, function=lambda δcα, δpβ: np.divide(δcα + δpβ, 2))


class GreeksEquation(StrategyEquation, metaclass=StrategyEquationMeta):
    vpα = Variable.Independent("vpα", "value", np.float32, locator=StrategyLocator(Securities.Options.Puts.Long, "value"))
    vpβ = Variable.Independent("vpβ", "value", np.float32, locator=StrategyLocator(Securities.Options.Puts.Short, "value"))
    vcα = Variable.Independent("vcα", "value", np.float32, locator=StrategyLocator(Securities.Options.Calls.Long, "value"))
    vcβ = Variable.Independent("vcβ", "value", np.float32, locator=StrategyLocator(Securities.Options.Calls.Short, "value"))

    Δpα = Variable.Independent("Δpα", "delta", np.float32, locator=StrategyLocator(Securities.Options.Puts.Long, "delta"))
    Δpβ = Variable.Independent("Δpβ", "delta", np.float32, locator=StrategyLocator(Securities.Options.Puts.Short, "delta"))
    Δcα = Variable.Independent("Δcα", "delta", np.float32, locator=StrategyLocator(Securities.Options.Calls.Long, "delta"))
    Δcβ = Variable.Independent("Δcβ", "delta", np.float32, locator=StrategyLocator(Securities.Options.Calls.Short, "delta"))

    Γpα = Variable.Independent("Γpα", "gamma", np.float32, locator=StrategyLocator(Securities.Options.Puts.Long, "gamma"))
    Γpβ = Variable.Independent("Γpβ", "gamma", np.float32, locator=StrategyLocator(Securities.Options.Puts.Short, "gamma"))
    Γcα = Variable.Independent("Γcα", "gamma", np.float32, locator=StrategyLocator(Securities.Options.Calls.Long, "gamma"))
    Γcβ = Variable.Independent("Γcβ", "gamma", np.float32, locator=StrategyLocator(Securities.Options.Calls.Short, "gamma"))

    Θpα = Variable.Independent("Θpα", "theta", np.float32, locator=StrategyLocator(Securities.Options.Puts.Long, "theta"))
    Θpβ = Variable.Independent("Θpβ", "theta", np.float32, locator=StrategyLocator(Securities.Options.Puts.Short, "theta"))
    Θcα = Variable.Independent("Θcα", "theta", np.float32, locator=StrategyLocator(Securities.Options.Calls.Long, "theta"))
    Θcβ = Variable.Independent("Θcβ", "theta", np.float32, locator=StrategyLocator(Securities.Options.Calls.Short, "theta"))

    Vpα = Variable.Independent("Vpα", "vega", np.float32, locator=StrategyLocator(Securities.Options.Puts.Long, "vega"))
    Vpβ = Variable.Independent("Vpβ", "vega", np.float32, locator=StrategyLocator(Securities.Options.Puts.Short, "vega"))
    Vcα = Variable.Independent("Vcα", "vega", np.float32, locator=StrategyLocator(Securities.Options.Calls.Long, "vega"))
    Vcβ = Variable.Independent("Vcβ", "vega", np.float32, locator=StrategyLocator(Securities.Options.Calls.Short, "vega"))

    Ppα = Variable.Independent("Ppα", "rho", np.float32, locator=StrategyLocator(Securities.Options.Puts.Long, "rho"))
    Ppβ = Variable.Independent("Ppβ", "rho", np.float32, locator=StrategyLocator(Securities.Options.Puts.Short, "rho"))
    Pcα = Variable.Independent("Pcα", "rho", np.float32, locator=StrategyLocator(Securities.Options.Calls.Long, "rho"))
    Pcβ = Variable.Independent("Pcβ", "rho", np.float32, locator=StrategyLocator(Securities.Options.Calls.Short, "rho"))

    def execute(self, *args, **kwargs):
        yield from super().execute(*args, **kwargs)
        yield self.vo()
        yield self.Δo()
        yield self.Γo()
        yield self.Θo()
        yield self.Vo()
        yield self.Po()


class VerticalPutGreeksEquation(GreeksEquation, register=Strategies.Verticals.Put):
    vo = Variable.Dependent("vo", "value", np.float32, function=lambda vpα, vpβ: vpα + vpβ)
    Δo = Variable.Dependent("Δo", "delta", np.float32, function=lambda Δpα, Δpβ: Δpα + Δpβ)
    Γo = Variable.Dependent("Γo", "gamma", np.float32, function=lambda Γpα, Γpβ: Γpα + Γpβ)
    Θo = Variable.Dependent("Θo", "theta", np.float32, function=lambda Θpα, Θpβ: Θpα + Θpβ)
    Vo = Variable.Dependent("Vo", "vega", np.float32, function=lambda Vpα, Vpβ: Vpα + Vpβ)
    Po = Variable.Dependent("Po", "rho", np.float32, function=lambda Ppα, Ppβ: Ppα + Ppβ)

class VerticalCallGreeksEquation(GreeksEquation, register=Strategies.Verticals.Call):
    vo = Variable.Dependent("vo", "value", np.float32, function=lambda vcα, vcβ: vcα + vcβ)
    Δo = Variable.Dependent("Δo", "delta", np.float32, function=lambda Δcα, Δcβ: Δcα + Δcβ)
    Γo = Variable.Dependent("Γo", "gamma", np.float32, function=lambda Γcα, Γcβ: Γcα + Γcβ)
    Θo = Variable.Dependent("Θo", "theta", np.float32, function=lambda Θcα, Θcβ: Θcα + Θcβ)
    Vo = Variable.Dependent("Vo", "vega", np.float32, function=lambda Vcα, Vcβ: Vcα + Vcβ)
    Po = Variable.Dependent("Po", "rho", np.float32, function=lambda Pcα, Pcβ: Pcα + Pcβ)

class CollarLongGreeksEquation(GreeksEquation, register=Strategies.Collars.Long):
    vo = Variable.Dependent("vo", "value", np.float32, function=lambda vpα, vcβ: vpα + vcβ)
    Δo = Variable.Dependent("Δo", "delta", np.float32, function=lambda Δpα, Δcβ: Δpα + Δcβ + 1)
    Γo = Variable.Dependent("Γo", "gamma", np.float32, function=lambda Γpα, Γcβ: Γpα + Γcβ)
    Θo = Variable.Dependent("Θo", "theta", np.float32, function=lambda Θpα, Θcβ: Θpα + Θcβ)
    Vo = Variable.Dependent("Vo", "vega", np.float32, function=lambda Vpα, Vcβ: Vpα + Vcβ)
    Po = Variable.Dependent("Po", "rho", np.float32, function=lambda Ppα, Pcβ: Ppα + Pcβ)

class CollarShortGreeksEquation(GreeksEquation, register=Strategies.Collars.Short):
    vo = Variable.Dependent("vo", "value", np.float32, function=lambda vcα, vpβ: vcα + vpβ)
    Δo = Variable.Dependent("Δo", "delta", np.float32, function=lambda Δcα, Δpβ: Δcα + Δpβ - 1)
    Γo = Variable.Dependent("Γo", "gamma", np.float32, function=lambda Γcα, Γpβ: Γcα + Γpβ)
    Θo = Variable.Dependent("Θo", "theta", np.float32, function=lambda Θcα, Θpβ: Θcα + Θpβ)
    Vo = Variable.Dependent("Vo", "vega", np.float32, function=lambda Vcα, Vpβ: Vcα + Vpβ)
    Po = Variable.Dependent("Po", "rho", np.float32, function=lambda Pcα, Ppβ: Pcα + Ppβ)


class StrategyCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, strategies, **kwargs):
        assert isinstance(strategies, list) and all([value in list(Strategies) for value in list(strategies)])
        super().__init__(*args, **kwargs)
        payoffs, underlying, greeks = dict(PayoffEquation), dict(UnderlyingEquation), dict(GreeksEquation)
        equations = {strategy: StrategyCollection(payoffs[strategy], underlying[strategy], greeks[strategy]) for strategy in strategies}
        equations = {strategy: dict(required=collection.payoff, optional=[collection.underlying, collection.greeks]) for strategy, collection in equations.items()}
        calculations = {strategy: Calculation[xr.DataArray](*args, **parameters, **kwargs) for strategy, parameters in equations.items()}
        self.__calculations = calculations

    def execute(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if self.empty(options): return
        generator = self.calculator(options, *args, **kwargs)
        for settlement, strategy, strategies in generator:
            size = self.size(strategies, "size")
            self.console(f"{str(settlement)}|{str(strategy)}[{int(size):.0f}]")
            if self.empty(strategies, "size"): return
            yield strategies

    def calculator(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        for settlement, dataframes in self.partition(options, by=Querys.Settlement):
            datasets = dict(self.unflatten(dataframes, *args, **kwargs))
            for strategy, calculation in self.calculations.items():
                if not all([option in datasets.keys() for option in strategy.options]): continue
                strategies = calculation(datasets, *args, **kwargs)
                assert isinstance(strategies, xr.Dataset)
                strategies = strategies.assign_coords({"strategy": xr.Variable("strategy", [strategy]).squeeze("strategy")})
                for field in list(Querys.Settlement): strategies = strategies.expand_dims(field)
                yield settlement, strategy, strategies

    @staticmethod
    def unflatten(options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
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




