# -*- coding: utf-8 -*-
"""
Created on Fri May 23 2025
@name:   Appraisal Objects
@author: Jack Kirby Cook

"""

import types
import numpy as np
import pandas as pd
from abc import ABC
from scipy.stats import norm

from finance.concepts import Concepts, Querys
from calculations import Equation, Variables, Algorithms, Computations
from support.mixins import Emptying, Sizing, Partition, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["AppraisalCalculator"]
__copyright__ = "Copyright 2025, Jack Kirby Cook"
__license__ = "MIT License"


class AppraisalEquation(Computations.Table, Algorithms.Vectorized.Table, Equation, ABC, root=True):
    zx = Variables.Dependent("zx", ("zscore", "itm"), np.float32, function=lambda zxk, zvt, zrt: zxk + zvt + zrt)
    zk = Variables.Dependent("zk", ("zscore", "otm"), np.float32, function=lambda zxk, zvt, zrt: zxk - zvt + zrt)

    zxk = Variables.Dependent("zxk", ("zscore", "strike"), np.float32, function=lambda x, k, σ, τ: np.log(x / k) / np.sqrt(τ) / σ)
    zvt = Variables.Dependent("zvt", ("zscore", "volatility"), np.float32, function=lambda σ, τ: np.sqrt(τ) * σ / 2)
    zrt = Variables.Dependent("zrt", ("zscore", "interest"), np.float32, function=lambda r, σ, τ: np.sqrt(τ) * r / σ)

    mkt = Variables.Independent("mkt", "price", np.float32, locator="price")
    tτ = Variables.Independent("tτ", "expire", np.datetime64, locator="expire")
    to = Variables.Constant("to", "current", np.datetime64, locator="current")

    i = Variables.Independent("i", "option", Concepts.Securities.Option, locator="option")
    j = Variables.Independent("j", "position", Concepts.Securities.Position, locator="position")
    x = Variables.Independent("x", "adjusted", np.float32, locator="adjusted")
    σ = Variables.Independent("σ", "volatility", np.float32, locator="volatility")
    μ = Variables.Independent("μ", "trend", np.float32, locator="trend")
    k = Variables.Independent("k", "strike", np.float32, locator="strike")
    r = Variables.Constant("r", "interest", np.float32, locator="interest")


class BlackScholesEquation(AppraisalEquation, ABC, register=Concepts.Appraisal.BLACKSCHOLES):
    y = Variables.Dependent("y", "value", np.float32, function=lambda x, k, zx, zk, r, τ, i: x * norm.cdf(zx * int(i)) * int(i) - k * norm.cdf(zk * int(i)) * int(i) / np.exp(r * τ))
    τ = Variables.Dependent("τ", "tau", np.float32, function=lambda to, tτ: (np.datetime64(tτ, "ns") - np.datetime64(to, "ns")) / np.timedelta64(364, 'D'))

    def execute(self, options, /, current, interest):
        parameters = dict(current=current, interest=interest)
        yield from super().execute(options, **parameters)
        yield self.y(options, **parameters)


class GreekEquation(AppraisalEquation, ABC, register=Concepts.Appraisal.GREEKS):
    Θ = Variables.Dependent("Θ", "theta", np.float32, function=lambda zx, zk, x, k, r, σ, τ, i: - norm.cdf(zk * int(i)) * int(i) * k * r / np.exp(r * τ) - norm.pdf(zx) * x * σ / np.sqrt(τ) / 2)
    P = Variables.Dependent("P", "rho", np.float32, function=lambda zk, k, r, τ, i: + norm.cdf(zk * int(i)) * int(i) * k * τ / np.exp(r * τ))
    Δ = Variables.Dependent("Δ", "delta", np.float32, function=lambda zx, i: + norm.cdf(zx * int(i)) * int(i))
    Γ = Variables.Dependent("Γ", "gamma", np.float32, function=lambda zx, x, σ, τ: + norm.pdf(zx) / np.sqrt(τ) / σ / x)
    V = Variables.Dependent("V", "vega", np.float32, function=lambda zx, x, τ: + norm.pdf(zx) * np.sqrt(τ) * x)
    τ = Variables.Dependent("τ", "tau", np.float32, function=lambda to, tτ: (np.datetime64(tτ, "ns") - np.datetime64(to, "ns")) / np.timedelta64(364, 'D'))

    def execute(self, options, /, current, interest):
        parameters = dict(current=current, interest=interest)
        yield from super().execute(options, **parameters)
        yield self.Θ(options, **parameters)
        yield self.P(options, **parameters)
        yield self.Δ(options, **parameters)
        yield self.Γ(options, **parameters)
        yield self.V(options, **parameters)


class AppraisalCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, appraisals, **kwargs):
        super().__init__(*args, **kwargs)
        equations = {appraisal: equation(*args, **kwargs) for appraisal, equation in iter(AppraisalEquation) if appraisal in appraisals}
        self.__equations = list(equations.values())

    def execute(self, options, technicals=None, /, **kwargs):
        assert isinstance(options, pd.DataFrame)
        assert isinstance(technicals, (pd.DataFrame, types.NoneType))
        if self.empty(options): return
        querys = self.keys(options, by=Querys.Settlement)
        querys = ",".join(list(map(str, querys)))
        if technicals is not None: options = self.technicals(options, technicals, **kwargs)
        options = self.calculate(options, **kwargs)
        size = self.size(options)
        self.console(f"{str(querys)}[{int(size):.0f}]")
        if self.empty(options): return
        yield options

    def calculate(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        appraisals = list(self.calculator(options, *args, **kwargs))
        results = pd.concat([options] + appraisals, axis=1)
        results = results.reset_index(drop=True, inplace=False)
        return results

    def calculator(self, options, *args, current, interest, **kwargs):
        assert isinstance(options, pd.DataFrame)
        for equation in self.equations:
            appraisals = equation(options, current=current, interest=interest)
            assert isinstance(appraisals, pd.DataFrame)
            yield appraisals

    @staticmethod
    def technicals(options, technicals, *args, **kwargs):
        assert isinstance(options, pd.DataFrame) and isinstance(technicals, pd.DataFrame)
        mask = technicals["date"] == technicals["date"].max()
        technicals = technicals.where(mask).dropna(how="all", inplace=False)
        technicals = technicals.drop(columns="date", inplace=False)
        options = options.merge(technicals, how="left", on=list(Querys.Symbol), sort=False, suffixes=("", ""))
        return options

    @property
    def equations(self): return self.__equations



