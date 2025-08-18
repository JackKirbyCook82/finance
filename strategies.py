# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Strategy Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
import xarray as xr
from enum import Enum
from abc import ABC, ABCMeta
from functools import reduce
from collections import namedtuple as ntuple

import calculations as calc
from finance.variables import Querys, Variables, Strategies, Securities
from support.mixins import Emptying, Sizing, Partition, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StrategyCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class StrategyLocator(ntuple("Locator", "axis security")): pass
class StrategyEquationMeta(type(calc.Equations.Array), ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, analytic=None, strategy=None, **kwargs):
        if bool(analytic): attrs = attrs | dict(analytic=analytic)
        if bool(strategy): attrs = attrs | dict(strategy=strategy)
        cls = super(StrategyEquationMeta, mcs).__new__(mcs, name, bases, attrs, *args, **kwargs)
        return cls

    def __iter__(cls): return iter(cls.registry.items())
    def __init__(cls, name, bases, attrs, *args, **kwargs):
        super(StrategyEquationMeta, cls).__init__(name, bases, attrs, *args, **kwargs)
        analytic = kwargs.get("analytic", getattr(cls, "analytic", None))
        strategy = kwargs.get("strategy", getattr(cls, "strategy", None))
        if not any([type(base) is StrategyEquationMeta for base in bases]):
            cls.__registry__ = dict()
        elif bool(analytic) and bool(strategy):
            cls.registry[strategy] = cls.registry.get(strategy, {})
            cls.registry[strategy][analytic] = cls

    @property
    def registry(cls): return cls.__registry__


class StrategyEquation(calc.Equations.Array, ABC, signature="[qpα,qpβ,qcα,qcβ,xpα,xpβ,xcα,xcβ,ypα,ypβ,ycα,ycβ]->[qo,xo,yo]", metaclass=StrategyEquationMeta):
    xpα = calc.Variables.Independent("xpα", "underlying", np.float32, locator=StrategyLocator(Securities.Options.Puts.Long, "underlying"))
    xpβ = calc.Variables.Independent("xpβ", "underlying", np.float32, locator=StrategyLocator(Securities.Options.Puts.Short, "underlying"))
    xcα = calc.Variables.Independent("xcα", "underlying", np.float32, locator=StrategyLocator(Securities.Options.Calls.Long, "underlying"))
    xcβ = calc.Variables.Independent("xcβ", "underlying", np.float32, locator=StrategyLocator(Securities.Options.Calls.Short, "underlying"))

    ypα = calc.Variables.Independent("ypα", "spot", np.float32, locator=StrategyLocator(Securities.Options.Puts.Long, "spot"))
    ypβ = calc.Variables.Independent("ypβ", "spot", np.float32, locator=StrategyLocator(Securities.Options.Puts.Short, "spot"))
    ycα = calc.Variables.Independent("ycα", "spot", np.float32, locator=StrategyLocator(Securities.Options.Calls.Long, "spot"))
    ycβ = calc.Variables.Independent("ycβ", "spot", np.float32, locator=StrategyLocator(Securities.Options.Calls.Short, "spot"))

    qpα = calc.Variables.Independent("qpα", "size", np.int32, locator=StrategyLocator(Securities.Options.Puts.Long, "size"))
    qpβ = calc.Variables.Independent("qpβ", "size", np.int32, locator=StrategyLocator(Securities.Options.Puts.Short, "size"))
    qcα = calc.Variables.Independent("qcα", "size", np.int32, locator=StrategyLocator(Securities.Options.Calls.Long, "size"))
    qcβ = calc.Variables.Independent("qcβ", "size", np.int32, locator=StrategyLocator(Securities.Options.Calls.Short, "size"))

#    def execute(self, *args, **kwargs):
#        yield from super().execute(*args, **kwargs)
#        yield self.qo()
#        yield self.xo()
#        yield self.yo()


class VerticalPutStrategyEquation(StrategyEquation, strategy=Strategies.Verticals.Put):
    xo = calc.Variables.Dependent("xo", "underlying", np.float32, function=lambda xpα, xpβ: np.divide(xpα + xpβ, 2))
    yo = calc.Variables.Dependent("yo", "spot", np.float32, function=lambda ypα, ypβ: ypβ + ypα)
    qo = calc.Variables.Dependent("qo", "size", np.int32, function=lambda qpα, qpβ: np.minimum(qpα, qpβ))

class VerticalCallStrategyEquation(StrategyEquation, strategy=Strategies.Verticals.Call):
    xo = calc.Variables.Dependent("xo", "underlying", np.float32, function=lambda xcα, xcβ: np.divide(xcα + xcβ, 2))
    yo = calc.Variables.Dependent("yo", "spot", np.float32, function=lambda ycα, ycβ: ycβ + ycα)
    qo = calc.Variables.Dependent("qo", "size", np.int32, function=lambda qcα, qcβ: np.minimum(qcα, qcβ))

class CollarLongStrategyEquation(StrategyEquation, strategy=Strategies.Collars.Long):
    xo = calc.Variables.Dependent("xo", "underlying", np.float32, function=lambda xpα, xcβ: np.divide(xpα + xcβ, 2))
    yo = calc.Variables.Dependent("yo", "spot", np.float32, function=lambda ypα, ycβ, xo: ycβ + ypα - xo)
    qo = calc.Variables.Dependent("qo", "size", np.int32, function=lambda qpα, qcβ: np.minimum(qpα, qcβ))

class CollarShortStrategyEquation(StrategyEquation, strategy=Strategies.Collars.Short):
    xo = calc.Variables.Dependent("xo", "underlying", np.float32, function=lambda xcα, xpβ: np.divide(xcα + xpβ, 2))
    yo = calc.Variables.Dependent("yo", "spot", np.float32, function=lambda ycα, ypβ, xo: ypβ + ycα + xo)
    qo = calc.Variables.Dependent("qo", "size", np.int32, function=lambda qcα, qpβ: np.minimum(qcα, qpβ))


class PayoffEquation(StrategyEquation, signature="[kpα,kpβ,kcα,kcβ]->[xl,xh,xk,yl,yh,yk]", analytic=Variables.Analytic.PAYOFF):
    mk = calc.Variables.Dependent("mk", "market", Enum, function=lambda kα, kβ: xr.where(kα < kβ, Variables.Market.BULL, xr.where(kα > kβ, Variables.Market.BEAR, Variables.Market.NEUTRAL)))
    yk = calc.Variables.Dependent("yk", "breakeven", np.float32, function=lambda yo, yl, yh: xr.where(np.negative(yo) <= yh, xr.where(np.negative(yo) >= yl, np.negative(yo), np.NaN), np.NaN))
    xk = calc.Variables.Dependent("xk", "pivot", np.float32, function=lambda yk, yl, yh, mk, xl: xl + (yk - yl) * np.abs(mk.astype(int) + 1) / 2 + (yh - yk) * np.abs(mk.astype(int) - 1) / 2)
    xh = calc.Variables.Dependent("xh", "lower", np.float32, function=lambda kα, kβ: np.maximum(kα, kβ))
    xl = calc.Variables.Dependent("xl", "higher", np.float32, function=lambda kα, kβ: np.minimum(kα, kβ))

    kpα = calc.Variables.Independent("kpα", "strike", np.float32, locator=StrategyLocator(Securities.Options.Puts.Long, "strike"))
    kpβ = calc.Variables.Independent("kpβ", "strike", np.float32, locator=StrategyLocator(Securities.Options.Puts.Short, "strike"))
    kcα = calc.Variables.Independent("kcα", "strike", np.float32, locator=StrategyLocator(Securities.Options.Calls.Long, "strike"))
    kcβ = calc.Variables.Independent("kcβ", "strike", np.float32, locator=StrategyLocator(Securities.Options.Calls.Short, "strike"))

#    def execute(self, *args, **kwargs):
#        yield from super().execute(*args, **kwargs)
#        yield self.yl()
#        yield self.yh()


class VerticalPutPayoffEquation(PayoffEquation, VerticalPutStrategyEquation):
    yh = calc.Variables.Dependent("yh", "maximum", np.float32, function=lambda kpα, kpβ: np.maximum(kpα - kpβ, 0))
    yl = calc.Variables.Dependent("yl", "minimum", np.float32, function=lambda kpα, kpβ: np.minimum(kpα - kpβ, 0))
    kα = calc.Variables.Dependent("kα", "strike", np.float32, function=lambda kpα: kpα)
    kβ = calc.Variables.Dependent("kβ", "strike", np.float32, function=lambda kpβ: kpβ)

class VerticalCallPayoffEquation(PayoffEquation, VerticalCallStrategyEquation):
    yh = calc.Variables.Dependent("yh", "maximum", np.float32, function=lambda kcα, kcβ: np.maximum(kcβ - kcα, 0))
    yl = calc.Variables.Dependent("yl", "minimum", np.float32, function=lambda kcα, kcβ: np.minimum(kcβ - kcα, 0))
    kα = calc.Variables.Dependent("kα", "strike", np.float32, function=lambda kcα: kcα)
    kβ = calc.Variables.Dependent("kβ", "strike", np.float32, function=lambda kcβ: kcβ)

class CollarLongPayoffEquation(PayoffEquation, CollarLongStrategyEquation):
    yh = calc.Variables.Dependent("yh", "maximum", np.float32, function=lambda kpα, kcβ: + np.maximum(kpα, kcβ))
    yl = calc.Variables.Dependent("yl", "minimum", np.float32, function=lambda kpα, kcβ: + np.minimum(kpα, kcβ))
    kα = calc.Variables.Dependent("kα", "strike", np.float32, function=lambda kpα: kpα)
    kβ = calc.Variables.Dependent("kβ", "strike", np.float32, function=lambda kcβ: kcβ)

class CollarShortPayoffEquation(PayoffEquation, CollarShortStrategyEquation):
    yh = calc.Variables.Dependent("yh", "maximum", np.float32, function=lambda kcα, kpβ: - np.minimum(kcα, kpβ))
    yl = calc.Variables.Dependent("yl", "minimum", np.float32, function=lambda kcα, kpβ: - np.maximum(kcα, kpβ))
    kα = calc.Variables.Dependent("kα", "strike", np.float32, function=lambda kcα: kcα)
    kβ = calc.Variables.Dependent("kβ", "strike", np.float32, function=lambda kpβ: kpβ)


class UnderlyingEquation(StrategyEquation, signature="[μpα,μpβ,μcα,μcβ,δpα,δpβ,δcα,δcβ]->[μo,δo]", analytic=Variables.Analytic.UNDERLYING):
    μpα = calc.Variables.Independent("μpα", "trend", np.float32, locator=StrategyLocator(Securities.Options.Puts.Long, "trend"))
    μpβ = calc.Variables.Independent("μpβ", "trend", np.float32, locator=StrategyLocator(Securities.Options.Puts.Short, "trend"))
    μcα = calc.Variables.Independent("μcα", "trend", np.float32, locator=StrategyLocator(Securities.Options.Calls.Long, "trend"))
    μcβ = calc.Variables.Independent("μcβ", "trend", np.float32, locator=StrategyLocator(Securities.Options.Calls.Short, "trend"))

    δpα = calc.Variables.Independent("δpα", "volatility", np.float32, locator=StrategyLocator(Securities.Options.Puts.Long, "volatility"))
    δpβ = calc.Variables.Independent("δpβ", "volatility", np.float32, locator=StrategyLocator(Securities.Options.Puts.Short, "volatility"))
    δcα = calc.Variables.Independent("δcα", "volatility", np.float32, locator=StrategyLocator(Securities.Options.Calls.Long, "volatility"))
    δcβ = calc.Variables.Independent("δcβ", "volatility", np.float32, locator=StrategyLocator(Securities.Options.Calls.Short, "volatility"))

#    def execute(self, *args, **kwargs):
#        yield from super().execute(*args, **kwargs)
#        yield self.μo()
#        yield self.δo()


class VerticalPutUnderlyingEquation(UnderlyingEquation, VerticalPutStrategyEquation):
    μo = calc.Variables.Dependent("μo", "trend", np.float32, function=lambda μpα, μpβ: np.divide(μpα + μpβ, 2))
    δo = calc.Variables.Dependent("δo", "volatility", np.float32, function=lambda δpα, δpβ: np.divide(δpα + δpβ, 2))

class VerticalCallUnderlyingEquation(UnderlyingEquation, VerticalCallStrategyEquation):
    μo = calc.Variables.Dependent("μo", "trend", np.float32, function=lambda μcα, μcβ: np.divide(μcα + μcβ, 2))
    δo = calc.Variables.Dependent("δo", "volatility", np.float32, function=lambda δcα, δcβ: np.divide(δcα + δcβ, 2))

class CollarLongUnderlyingEquation(UnderlyingEquation, CollarLongStrategyEquation):
    μo = calc.Variables.Dependent("μo", "trend", np.float32, function=lambda μpα, μcβ: np.divide(μpα + μcβ, 2))
    δo = calc.Variables.Dependent("δo", "volatility", np.float32, function=lambda δpα, δcβ: np.divide(δpα + δcβ, 2))

class CollarShortUnderlyingEquation(UnderlyingEquation, CollarShortStrategyEquation):
    μo = calc.Variables.Dependent("μo", "trend", np.float32, function=lambda μcα, μpβ: np.divide(μcα + μpβ, 2))
    δo = calc.Variables.Dependent("δo", "volatility", np.float32, function=lambda δcα, δpβ: np.divide(δcα + δpβ, 2))


class GreeksEquation(StrategyEquation, signature="[vpα,vpβ,vcα,vcβ,Δpα,Δpβ,Δcα,Δcβ,Γpα,Γpβ,Γcα,Γcβ,Θpα,Θpβ,Θcα,Θcβ,Vpα,Vpβ,Vcα,Vcβ]->[vo,Δo,Γo,Θo,Vo]", analytic=Variables.Analytic.GREEKS):
    vpα = calc.Variables.Independent("vpα", "value", np.float32, locator=StrategyLocator(Securities.Options.Puts.Long, "value"))
    vpβ = calc.Variables.Independent("vpβ", "value", np.float32, locator=StrategyLocator(Securities.Options.Puts.Short, "value"))
    vcα = calc.Variables.Independent("vcα", "value", np.float32, locator=StrategyLocator(Securities.Options.Calls.Long, "value"))
    vcβ = calc.Variables.Independent("vcβ", "value", np.float32, locator=StrategyLocator(Securities.Options.Calls.Short, "value"))

    Δpα = calc.Variables.Independent("Δpα", "delta", np.float32, locator=StrategyLocator(Securities.Options.Puts.Long, "delta"))
    Δpβ = calc.Variables.Independent("Δpβ", "delta", np.float32, locator=StrategyLocator(Securities.Options.Puts.Short, "delta"))
    Δcα = calc.Variables.Independent("Δcα", "delta", np.float32, locator=StrategyLocator(Securities.Options.Calls.Long, "delta"))
    Δcβ = calc.Variables.Independent("Δcβ", "delta", np.float32, locator=StrategyLocator(Securities.Options.Calls.Short, "delta"))

    Γpα = calc.Variables.Independent("Γpα", "gamma", np.float32, locator=StrategyLocator(Securities.Options.Puts.Long, "gamma"))
    Γpβ = calc.Variables.Independent("Γpβ", "gamma", np.float32, locator=StrategyLocator(Securities.Options.Puts.Short, "gamma"))
    Γcα = calc.Variables.Independent("Γcα", "gamma", np.float32, locator=StrategyLocator(Securities.Options.Calls.Long, "gamma"))
    Γcβ = calc.Variables.Independent("Γcβ", "gamma", np.float32, locator=StrategyLocator(Securities.Options.Calls.Short, "gamma"))

    Θpα = calc.Variables.Independent("Θpα", "theta", np.float32, locator=StrategyLocator(Securities.Options.Puts.Long, "theta"))
    Θpβ = calc.Variables.Independent("Θpβ", "theta", np.float32, locator=StrategyLocator(Securities.Options.Puts.Short, "theta"))
    Θcα = calc.Variables.Independent("Θcα", "theta", np.float32, locator=StrategyLocator(Securities.Options.Calls.Long, "theta"))
    Θcβ = calc.Variables.Independent("Θcβ", "theta", np.float32, locator=StrategyLocator(Securities.Options.Calls.Short, "theta"))

    Vpα = calc.Variables.Independent("Vpα", "vega", np.float32, locator=StrategyLocator(Securities.Options.Puts.Long, "vega"))
    Vpβ = calc.Variables.Independent("Vpβ", "vega", np.float32, locator=StrategyLocator(Securities.Options.Puts.Short, "vega"))
    Vcα = calc.Variables.Independent("Vcα", "vega", np.float32, locator=StrategyLocator(Securities.Options.Calls.Long, "vega"))
    Vcβ = calc.Variables.Independent("Vcβ", "vega", np.float32, locator=StrategyLocator(Securities.Options.Calls.Short, "vega"))

#    def execute(self, *args, **kwargs):
#        yield from super().execute(*args, **kwargs)
#        yield self.vo()
#        yield self.Δo()
#        yield self.Γo()
#        yield self.Θo()
#        yield self.Vo()
#        yield self.Po()


class VerticalPutGreeksEquation(GreeksEquation, VerticalPutStrategyEquation):
    vo = calc.Variables.Dependent("vo", "value", np.float32, function=lambda vpα, vpβ: vpα + vpβ)
    Δo = calc.Variables.Dependent("Δo", "delta", np.float32, function=lambda Δpα, Δpβ: Δpα + Δpβ)
    Γo = calc.Variables.Dependent("Γo", "gamma", np.float32, function=lambda Γpα, Γpβ: Γpα + Γpβ)
    Θo = calc.Variables.Dependent("Θo", "theta", np.float32, function=lambda Θpα, Θpβ: Θpα + Θpβ)
    Vo = calc.Variables.Dependent("Vo", "vega", np.float32, function=lambda Vpα, Vpβ: Vpα + Vpβ)

class VerticalCallGreeksEquation(GreeksEquation, VerticalCallStrategyEquation):
    vo = calc.Variables.Dependent("vo", "value", np.float32, function=lambda vcα, vcβ: vcα + vcβ)
    Δo = calc.Variables.Dependent("Δo", "delta", np.float32, function=lambda Δcα, Δcβ: Δcα + Δcβ)
    Γo = calc.Variables.Dependent("Γo", "gamma", np.float32, function=lambda Γcα, Γcβ: Γcα + Γcβ)
    Θo = calc.Variables.Dependent("Θo", "theta", np.float32, function=lambda Θcα, Θcβ: Θcα + Θcβ)
    Vo = calc.Variables.Dependent("Vo", "vega", np.float32, function=lambda Vcα, Vcβ: Vcα + Vcβ)

class CollarLongGreeksEquation(GreeksEquation, CollarLongStrategyEquation):
    vo = calc.Variables.Dependent("vo", "value", np.float32, function=lambda vpα, vcβ: vpα + vcβ)
    Δo = calc.Variables.Dependent("Δo", "delta", np.float32, function=lambda Δpα, Δcβ: Δpα + Δcβ + 1)
    Γo = calc.Variables.Dependent("Γo", "gamma", np.float32, function=lambda Γpα, Γcβ: Γpα + Γcβ)
    Θo = calc.Variables.Dependent("Θo", "theta", np.float32, function=lambda Θpα, Θcβ: Θpα + Θcβ)
    Vo = calc.Variables.Dependent("Vo", "vega", np.float32, function=lambda Vpα, Vcβ: Vpα + Vcβ)

class CollarShortGreeksEquation(GreeksEquation, CollarShortStrategyEquation):
    vo = calc.Variables.Dependent("vo", "value", np.float32, function=lambda vcα, vpβ: vcα + vpβ)
    Δo = calc.Variables.Dependent("Δo", "delta", np.float32, function=lambda Δcα, Δpβ: Δcα + Δpβ - 1)
    Γo = calc.Variables.Dependent("Γo", "gamma", np.float32, function=lambda Γcα, Γpβ: Γcα + Γpβ)
    Θo = calc.Variables.Dependent("Θo", "theta", np.float32, function=lambda Θcα, Θpβ: Θcα + Θpβ)
    Vo = calc.Variables.Dependent("Vo", "vega", np.float32, function=lambda Vcα, Vpβ: Vcα + Vpβ)


class StrategyCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, strategies, analytics, **kwargs):
        assert isinstance(strategies, list) and all([value in list(Strategies) for value in list(strategies)])
        assert isinstance(analytics, list) and all([value in list(Variables.Analytic) for value in list(analytics)])
        super().__init__(*args, **kwargs)
        equations = {strategy: equations for strategy, equations in iter(StrategyEquation) if strategy in strategies}
        equations = {strategy: [equation for analytic, equation in equations.items() if analytic in analytics] for strategy, equations in equations.items()}
        equations = {strategy: StrategyEquationMeta("Equation", tuple(equations), {"strategy": strategy, "analytic": None}) for strategy, equations in equations.items()}
        calculations = {strategy: Calculation[xr.DataArray](*args, equation=equation, **kwargs) for strategy, equation in equations.items()}
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




