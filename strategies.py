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

from finance.concepts import Querys, Concepts, Strategies, Securities
from calculations import Variables, Equations, DomainError
from support.mixins import Emptying, Sizing, Partition, Logging
from support.decorators import Signature

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StrategyCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class StrategyLocator(ntuple("Locator", "axis security")): pass
class StrategyEquationMeta(type(Equations.UnVectorized.Array), ABCMeta):
    def __init__(cls, name, bases, attrs, *args, strategy=None, **kwargs):
        super(StrategyEquationMeta, cls).__init__(name, bases, attrs)
        if not any([type(base) is StrategyEquationMeta for base in bases]):
            cls.__registry__ = dict()
            cls.__strategy__ = None
        elif cls.strategy is None and strategy is None: pass
        elif cls.strategy is None and strategy is not None:
            cls.registry[strategy] = list()
            cls.strategy = strategy
        elif cls.strategy is not None and strategy is None:
            cls.registry[cls.strategy].append(cls)
        else: raise ValueError(strategy)

    @property
    def registry(cls): return cls.__registry__
    @property
    def strategy(cls): return cls.__strategy__
    @strategy.setter
    def strategy(cls, strategy): cls.__strategy__ = strategy


class StrategyEquation(Equations.UnVectorized.Array, ABC, metaclass=StrategyEquationMeta):
    xpα = Variables.Independent("xpα", ("put", "long", "underlying"), np.float32, locator=StrategyLocator(Securities.Options.Puts.Long, "underlying"))
    xpβ = Variables.Independent("xpβ", ("put", "short", "underlying"), np.float32, locator=StrategyLocator(Securities.Options.Puts.Short, "underlying"))
    xcα = Variables.Independent("xcα", ("call", "long", "underlying"), np.float32, locator=StrategyLocator(Securities.Options.Calls.Long, "underlying"))
    xcβ = Variables.Independent("xcβ", ("call", "short", "underlying"), np.float32, locator=StrategyLocator(Securities.Options.Calls.Short, "underlying"))

    ypα = Variables.Independent("ypα", ("put", "long", "spot"), np.float32, locator=StrategyLocator(Securities.Options.Puts.Long, "spot"))
    ypβ = Variables.Independent("ypβ", ("put", "short", "spot"), np.float32, locator=StrategyLocator(Securities.Options.Puts.Short, "spot"))
    ycα = Variables.Independent("ycα", ("call", "long", "spot"), np.float32, locator=StrategyLocator(Securities.Options.Calls.Long, "spot"))
    ycβ = Variables.Independent("ycβ", ("call", "short", "spot"), np.float32, locator=StrategyLocator(Securities.Options.Calls.Short, "spot"))

    qpα = Variables.Independent("qpα", ("put", "long", "size"), np.int32, locator=StrategyLocator(Securities.Options.Puts.Long, "size"))
    qpβ = Variables.Independent("qpβ", ("put", "short", "size"), np.int32, locator=StrategyLocator(Securities.Options.Puts.Short, "size"))
    qcα = Variables.Independent("qcα", ("call", "long", "size"), np.int32, locator=StrategyLocator(Securities.Options.Calls.Long, "size"))
    qcβ = Variables.Independent("qcβ", ("call", "short", "size"), np.int32, locator=StrategyLocator(Securities.Options.Calls.Short, "size"))

    def execute(self, *args, **kwargs):
        yield from super().execute(*args, **kwargs)
        yield self.xo()
        yield self.yo()
        yield self.qo()


class VerticalPutStrategyEquation(StrategyEquation, strategy=Strategies.Verticals.Put):
    xo = Variables.Dependent("xo", "underlying", np.float32, function=lambda xpα, xpβ: np.divide(xpα + xpβ, 2))
    yo = Variables.Dependent("yo", "spot", np.float32, function=lambda ypα, ypβ: ypβ + ypα)
    qo = Variables.Dependent("qo", "size", np.int32, function=lambda qpα, qpβ: np.minimum(qpα, qpβ))

class VerticalCallStrategyEquation(StrategyEquation, strategy=Strategies.Verticals.Call):
    xo = Variables.Dependent("xo", "underlying", np.float32, function=lambda xcα, xcβ: np.divide(xcα + xcβ, 2))
    yo = Variables.Dependent("yo", "spot", np.float32, function=lambda ycα, ycβ: ycβ + ycα)
    qo = Variables.Dependent("qo", "size", np.int32, function=lambda qcα, qcβ: np.minimum(qcα, qcβ))

class CollarLongStrategyEquation(StrategyEquation, strategy=Strategies.Collars.Long):
    xo = Variables.Dependent("xo", "underlying", np.float32, function=lambda xpα, xcβ: np.divide(xpα + xcβ, 2))
    yo = Variables.Dependent("yo", "spot", np.float32, function=lambda ypα, ycβ, xo: ycβ + ypα - xo)
    qo = Variables.Dependent("qo", "size", np.int32, function=lambda qpα, qcβ: np.minimum(qpα, qcβ))

class CollarShortStrategyEquation(StrategyEquation, strategy=Strategies.Collars.Short):
    xo = Variables.Dependent("xo", "underlying", np.float32, function=lambda xcα, xpβ: np.divide(xcα + xpβ, 2))
    yo = Variables.Dependent("yo", "spot", np.float32, function=lambda ycα, ypβ, xo: ypβ + ycα + xo)
    qo = Variables.Dependent("qo", "size", np.int32, function=lambda qcα, qpβ: np.minimum(qcα, qpβ))


class PayoffEquation(StrategyEquation):
    mk = Variables.Dependent("mk", "market", Enum, function=lambda kα, kβ: xr.where(kα < kβ, Concepts.Market.BULL, xr.where(kα > kβ, Concepts.Market.BEAR, Concepts.Market.NEUTRAL)))
    yk = Variables.Dependent("yk", "breakeven", np.float32, function=lambda yo, yl, yh: xr.where(np.negative(yo) <= yh, xr.where(np.negative(yo) >= yl, np.negative(yo), np.NaN), np.NaN))
    xk = Variables.Dependent("xk", "pivot", np.float32, function=lambda yk, yl, yh, mk, xl: xl + (yk - yl) * np.abs(mk.astype(int) + 1) / 2 + (yh - yk) * np.abs(mk.astype(int) - 1) / 2)
    xh = Variables.Dependent("xh", "higher", np.float32, function=lambda kα, kβ: np.maximum(kα, kβ))
    xl = Variables.Dependent("xl", "lower", np.float32, function=lambda kα, kβ: np.minimum(kα, kβ))

    kpα = Variables.Independent("kpα", ("put", "long", "strike"), np.float32, locator=StrategyLocator(Securities.Options.Puts.Long, "strike"))
    kpβ = Variables.Independent("kpβ", ("put", "short", "strike"), np.float32, locator=StrategyLocator(Securities.Options.Puts.Short, "strike"))
    kcα = Variables.Independent("kcα", ("call", "long", "strike"), np.float32, locator=StrategyLocator(Securities.Options.Calls.Long, "strike"))
    kcβ = Variables.Independent("kcβ", ("call", "short", "strike"), np.float32, locator=StrategyLocator(Securities.Options.Calls.Short, "strike"))

    def execute(self, *args, **kwargs):
        yield from super().execute(*args, **kwargs)
        yield self.yh()
        yield self.yk()
        yield self.yl()


class VerticalPutPayoffEquation(PayoffEquation, VerticalPutStrategyEquation):
    yh = Variables.Dependent("yh", "maximum", np.float32, function=lambda kpα, kpβ: np.maximum(kpα - kpβ, 0))
    yl = Variables.Dependent("yl", "minimum", np.float32, function=lambda kpα, kpβ: np.minimum(kpα - kpβ, 0))
    kα = Variables.Dependent("kα", ("long", "strike"), np.float32, function=lambda kpα: kpα)
    kβ = Variables.Dependent("kβ", ("short", "strike"), np.float32, function=lambda kpβ: kpβ)

class VerticalCallPayoffEquation(PayoffEquation, VerticalCallStrategyEquation):
    yh = Variables.Dependent("yh", "maximum", np.float32, function=lambda kcα, kcβ: np.maximum(kcβ - kcα, 0))
    yl = Variables.Dependent("yl", "minimum", np.float32, function=lambda kcα, kcβ: np.minimum(kcβ - kcα, 0))
    kα = Variables.Dependent("kα", ("long", "strike"), np.float32, function=lambda kcα: kcα)
    kβ = Variables.Dependent("kβ", ("short", "strike"), np.float32, function=lambda kcβ: kcβ)

class CollarLongPayoffEquation(PayoffEquation, CollarLongStrategyEquation):
    yh = Variables.Dependent("yh", "maximum", np.float32, function=lambda kpα, kcβ: + np.maximum(kpα, kcβ))
    yl = Variables.Dependent("yl", "minimum", np.float32, function=lambda kpα, kcβ: + np.minimum(kpα, kcβ))
    kα = Variables.Dependent("kα", ("long", "strike"), np.float32, function=lambda kpα: kpα)
    kβ = Variables.Dependent("kβ", ("short", "strike"), np.float32, function=lambda kcβ: kcβ)

class CollarShortPayoffEquation(PayoffEquation, CollarShortStrategyEquation):
    yh = Variables.Dependent("yh", "maximum", np.float32, function=lambda kcα, kpβ: - np.minimum(kcα, kpβ))
    yl = Variables.Dependent("yl", "minimum", np.float32, function=lambda kcα, kpβ: - np.maximum(kcα, kpβ))
    kα = Variables.Dependent("kα", ("long", "strike"), np.float32, function=lambda kcα: kcα)
    kβ = Variables.Dependent("kβ", ("short", "strike"), np.float32, function=lambda kpβ: kpβ)


class UnderlyingEquation(StrategyEquation):
    μpα = Variables.Independent("μpα", ("put", "long", "trend"), np.float32, locator=StrategyLocator(Securities.Options.Puts.Long, "trend"))
    μpβ = Variables.Independent("μpβ", ("put", "short", "trend"), np.float32, locator=StrategyLocator(Securities.Options.Puts.Short, "trend"))
    μcα = Variables.Independent("μcα", ("call", "long", "trend"), np.float32, locator=StrategyLocator(Securities.Options.Calls.Long, "trend"))
    μcβ = Variables.Independent("μcβ", ("call", "short", "trend"), np.float32, locator=StrategyLocator(Securities.Options.Calls.Short, "trend"))

    δpα = Variables.Independent("δpα", ("put", "long", "volatility"), np.float32, locator=StrategyLocator(Securities.Options.Puts.Long, "volatility"))
    δpβ = Variables.Independent("δpβ", ("put", "short", "volatility"), np.float32, locator=StrategyLocator(Securities.Options.Puts.Short, "volatility"))
    δcα = Variables.Independent("δcα", ("call", "long", "volatility"), np.float32, locator=StrategyLocator(Securities.Options.Calls.Long, "volatility"))
    δcβ = Variables.Independent("δcβ", ("call", "short", "volatility"), np.float32, locator=StrategyLocator(Securities.Options.Calls.Short, "volatility"))

    def execute(self, *args, **kwargs):
        yield from super().execute(*args, **kwargs)
        for attribute in str("μo,δo").split(","):
            try: getattr(self, attribute)()
            except DomainError: pass


class VerticalPutUnderlyingEquation(UnderlyingEquation, VerticalPutStrategyEquation):
    μo = Variables.Dependent("μo", "trend", np.float32, function=lambda μpα, μpβ: np.divide(μpα + μpβ, 2))
    δo = Variables.Dependent("δo", "volatility", np.float32, function=lambda δpα, δpβ: np.divide(δpα + δpβ, 2))

class VerticalCallUnderlyingEquation(UnderlyingEquation, VerticalCallStrategyEquation):
    μo = Variables.Dependent("μo", "trend", np.float32, function=lambda μcα, μcβ: np.divide(μcα + μcβ, 2))
    δo = Variables.Dependent("δo", "volatility", np.float32, function=lambda δcα, δcβ: np.divide(δcα + δcβ, 2))

class CollarLongUnderlyingEquation(UnderlyingEquation, CollarLongStrategyEquation):
    μo = Variables.Dependent("μo", "trend", np.float32, function=lambda μpα, μcβ: np.divide(μpα + μcβ, 2))
    δo = Variables.Dependent("δo", "volatility", np.float32, function=lambda δpα, δcβ: np.divide(δpα + δcβ, 2))

class CollarShortUnderlyingEquation(UnderlyingEquation, CollarShortStrategyEquation):
    μo = Variables.Dependent("μo", "trend", np.float32, function=lambda μcα, μpβ: np.divide(μcα + μpβ, 2))
    δo = Variables.Dependent("δo", "volatility", np.float32, function=lambda δcα, δpβ: np.divide(δcα + δpβ, 2))


class AppraisalEquation(StrategyEquation):
    vpα = Variables.Independent("vpα", ("put", "long", "value"), np.float32, locator=StrategyLocator(Securities.Options.Puts.Long, "value"))
    vpβ = Variables.Independent("vpβ", ("put", "short", "value"), np.float32, locator=StrategyLocator(Securities.Options.Puts.Short, "value"))
    vcα = Variables.Independent("vcα", ("call", "long", "value"), np.float32, locator=StrategyLocator(Securities.Options.Calls.Long, "value"))
    vcβ = Variables.Independent("vcβ", ("call", "short", "value"), np.float32, locator=StrategyLocator(Securities.Options.Calls.Short, "value"))

    Δpα = Variables.Independent("Δpα", ("put", "long", "delta"), np.float32, locator=StrategyLocator(Securities.Options.Puts.Long, "delta"))
    Δpβ = Variables.Independent("Δpβ", ("put", "short", "delta"), np.float32, locator=StrategyLocator(Securities.Options.Puts.Short, "delta"))
    Δcα = Variables.Independent("Δcα", ("call", "long", "delta"), np.float32, locator=StrategyLocator(Securities.Options.Calls.Long, "delta"))
    Δcβ = Variables.Independent("Δcβ", ("call", "short", "delta"), np.float32, locator=StrategyLocator(Securities.Options.Calls.Short, "delta"))

    Γpα = Variables.Independent("Γpα", ("put", "long", "gamma"), np.float32, locator=StrategyLocator(Securities.Options.Puts.Long, "gamma"))
    Γpβ = Variables.Independent("Γpβ", ("put", "short", "gamma"), np.float32, locator=StrategyLocator(Securities.Options.Puts.Short, "gamma"))
    Γcα = Variables.Independent("Γcα", ("call", "long", "gamma"), np.float32, locator=StrategyLocator(Securities.Options.Calls.Long, "gamma"))
    Γcβ = Variables.Independent("Γcβ", ("call", "short", "gamma"), np.float32, locator=StrategyLocator(Securities.Options.Calls.Short, "gamma"))

    Θpα = Variables.Independent("Θpα", ("put", "long", "theta"), np.float32, locator=StrategyLocator(Securities.Options.Puts.Long, "theta"))
    Θpβ = Variables.Independent("Θpβ", ("put", "short", "theta"), np.float32, locator=StrategyLocator(Securities.Options.Puts.Short, "theta"))
    Θcα = Variables.Independent("Θcα", ("call", "long", "theta"), np.float32, locator=StrategyLocator(Securities.Options.Calls.Long, "theta"))
    Θcβ = Variables.Independent("Θcβ", ("call", "short", "theta"), np.float32, locator=StrategyLocator(Securities.Options.Calls.Short, "theta"))

    Vpα = Variables.Independent("Vpα", ("put", "long", "vega"), np.float32, locator=StrategyLocator(Securities.Options.Puts.Long, "vega"))
    Vpβ = Variables.Independent("Vpβ", ("put", "short", "vega"), np.float32, locator=StrategyLocator(Securities.Options.Puts.Short, "vega"))
    Vcα = Variables.Independent("Vcα", ("call", "long", "vega"), np.float32, locator=StrategyLocator(Securities.Options.Calls.Long, "vega"))
    Vcβ = Variables.Independent("Vcβ", ("call", "short", "vega"), np.float32, locator=StrategyLocator(Securities.Options.Calls.Short, "vega"))

    def execute(self, *args, **kwargs):
        yield from super().execute(*args, **kwargs)
        for attribute in str("vo,Δo,Γo,Θo,Vo").split(","):
            try: getattr(self, attribute)()
            except DomainError: pass


class VerticalPutAppraisalEquation(AppraisalEquation, VerticalPutStrategyEquation):
    vo = Variables.Dependent("vo", "value", np.float32, function=lambda vpα, vpβ: vpα + vpβ)
    Δo = Variables.Dependent("Δo", "delta", np.float32, function=lambda Δpα, Δpβ: Δpα + Δpβ)
    Γo = Variables.Dependent("Γo", "gamma", np.float32, function=lambda Γpα, Γpβ: Γpα + Γpβ)
    Θo = Variables.Dependent("Θo", "theta", np.float32, function=lambda Θpα, Θpβ: Θpα + Θpβ)
    Vo = Variables.Dependent("Vo", "vega", np.float32, function=lambda Vpα, Vpβ: Vpα + Vpβ)

class VerticalCallAppraisalEquation(AppraisalEquation, VerticalCallStrategyEquation):
    vo = Variables.Dependent("vo", "value", np.float32, function=lambda vcα, vcβ: vcα + vcβ)
    Δo = Variables.Dependent("Δo", "delta", np.float32, function=lambda Δcα, Δcβ: Δcα + Δcβ)
    Γo = Variables.Dependent("Γo", "gamma", np.float32, function=lambda Γcα, Γcβ: Γcα + Γcβ)
    Θo = Variables.Dependent("Θo", "theta", np.float32, function=lambda Θcα, Θcβ: Θcα + Θcβ)
    Vo = Variables.Dependent("Vo", "vega", np.float32, function=lambda Vcα, Vcβ: Vcα + Vcβ)

class CollarLongAppraisalEquation(AppraisalEquation, CollarLongStrategyEquation):
    vo = Variables.Dependent("vo", "value", np.float32, function=lambda vpα, vcβ: vpα + vcβ)
    Δo = Variables.Dependent("Δo", "delta", np.float32, function=lambda Δpα, Δcβ: Δpα + Δcβ + 1)
    Γo = Variables.Dependent("Γo", "gamma", np.float32, function=lambda Γpα, Γcβ: Γpα + Γcβ)
    Θo = Variables.Dependent("Θo", "theta", np.float32, function=lambda Θpα, Θcβ: Θpα + Θcβ)
    Vo = Variables.Dependent("Vo", "vega", np.float32, function=lambda Vpα, Vcβ: Vpα + Vcβ)

class CollarShortAppraisalEquation(AppraisalEquation, CollarShortStrategyEquation):
    vo = Variables.Dependent("vo", "value", np.float32, function=lambda vcα, vpβ: vcα + vpβ)
    Δo = Variables.Dependent("Δo", "delta", np.float32, function=lambda Δcα, Δpβ: Δcα + Δpβ - 1)
    Γo = Variables.Dependent("Γo", "gamma", np.float32, function=lambda Γcα, Γpβ: Γcα + Γpβ)
    Θo = Variables.Dependent("Θo", "theta", np.float32, function=lambda Θcα, Θpβ: Θcα + Θpβ)
    Vo = Variables.Dependent("Vo", "vega", np.float32, function=lambda Vcα, Vpβ: Vcα + Vpβ)


class StrategyCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, strategies, **kwargs):
        assert isinstance(strategies, list) and all([value in list(Strategies) for value in list(strategies)])
        super().__init__(*args, **kwargs)
        equations = {strategy: list(StrategyEquation.registry[strategy]) for strategy in strategies}
        equations = {key: StrategyEquation + list(values) for key, values in equations.items()}
        self.__equations = equations

    @Signature("options->strategies")
    def execute(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if self.empty(options): return

        print(options)
        raise Exception()

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
            for strategy, equation in self.equations.items():
                if not all([option in datasets.keys() for option in strategy.options]): continue
                equation = equation(arguments=datasets, parameters={})
                strategies = equation(*args, **kwargs)
                assert isinstance(strategies, xr.Dataset)
                strategies = strategies.assign_coords({"strategy": xr.Variable("strategy", [strategy]).squeeze("strategy")})
                for field in list(Querys.Settlement): strategies = strategies.expand_dims(field)
                yield settlement, strategy, strategies

    @staticmethod
    def unflatten(options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        for security, dataframe in options.groupby(list(Concepts.Securities.Security), sort=False):
            if dataframe.empty: continue
            security = Securities(security)
            dataframe = dataframe.drop(columns=list(Concepts.Securities.Security))
            dataframe = dataframe.set_index(list(Querys.Settlement) + ["strike"], drop=True, inplace=False)
            dataset = xr.Dataset.from_dataframe(dataframe)
            dataset = reduce(lambda content, axis: content.squeeze(axis), list(Querys.Settlement), dataset)
            dataset = dataset.rename({"strike": str(security)})
            dataset["strike"] = dataset[str(security)]
            yield security, dataset

    @property
    def equations(self): return self.__equations



