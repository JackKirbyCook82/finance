# -*- coding: utf-8 -*-
"""
Created on Fri May 23 2025
@name:   Appraisal Objects
@author: Jack Kirby Cook

"""

import types
import numpy as np
import pandas as pd
from abc import ABC, ABCMeta
from scipy.stats import norm

from finance.concepts import Concepts, Querys
from calculations import Equation, Variables, Algorithms, Computations
from support.mixins import Emptying, Sizing, Partition, Logging
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["AppraisalCalculator"]
__copyright__ = "Copyright 2025, Jack Kirby Cook"
__license__ = "MIT License"


class AppraisalEquationMeta(RegistryMeta, type(Equation), ABCMeta): pass
class AppraisalEquation(Computations.Table, Algorithms.Vectorized.Table, Equation, ABC, metaclass=AppraisalEquationMeta):
    τ = Variables.Dependent("τ", "tau", np.float32, function=lambda to, tτ: (np.datetime64(tτ, "ns") - np.datetime64(to, "ns")) / np.timedelta64(364, 'D'))

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

    def execute(self, contents, /, current, interest):
        parameters = dict(current=current, interest=interest)
        yield from super().execute(contents, **parameters)
        yield self.y(contents, **parameters)


class GreekEquation(AppraisalEquation, ABC, register=Concepts.Appraisal.GREEKS):
    Θ = Variables.Dependent("Θ", "theta", np.float32, function=lambda zx, zk, x, k, r, σ, τ, i: - norm.cdf(zk * int(i)) * int(i) * k * r / np.exp(r * τ) - norm.pdf(zx) * x * σ / np.sqrt(τ) / 2)
    P = Variables.Dependent("P", "rho", np.float32, function=lambda zk, k, r, τ, i: + norm.cdf(zk * int(i)) * int(i) * k * τ / np.exp(r * τ))
    Δ = Variables.Dependent("Δ", "delta", np.float32, function=lambda zx, i: + norm.cdf(zx * int(i)) * int(i))
    Γ = Variables.Dependent("Γ", "gamma", np.float32, function=lambda zx, x, σ, τ: + norm.pdf(zx) / np.sqrt(τ) / σ / x)
    V = Variables.Dependent("V", "vega", np.float32, function=lambda zx, x, τ: + norm.pdf(zx) * np.sqrt(τ) * x)

    def execute(self, contents, /, current, interest):
        parameters = dict(current=current, interest=interest)
        yield from super().execute(contents, **parameters)
        yield self.Θ(contents, **parameters)
        yield self.P(contents, **parameters)
        yield self.Δ(contents, **parameters)
        yield self.Γ(contents, **parameters)
        yield self.V(contents, **parameters)


class AppraisalCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, appraisals, **kwargs):
        assert isinstance(appraisals, list) and all([value in list(Concepts.Appraisal) for value in list(appraisals)])
        super().__init__(*args, **kwargs)
        self.__equations = {appraisal: equation(*args, **kwargs) for appraisal, equation in iter(AppraisalEquation) if appraisal in appraisals}

    def execute(self, contents, /, technicals=None, **kwargs):
        assert isinstance(contents, pd.DataFrame)
        assert isinstance(technicals, (pd.DataFrame, types.NoneType))
        if self.empty(contents): return
        querys = self.keys(contents, by=Querys.Settlement)
        querys = ",".join(list(map(str, querys)))
        if technicals is not None: contents = self.technicals(contents, technicals, **kwargs)
        results = self.calculate(contents, **kwargs)
        size = self.size(results)
        self.console(f"{str(querys)}[{int(size):.0f}]")
        if self.empty(results): return
        yield results

    def calculate(self, contents, /, **kwargs):
        assert isinstance(contents, pd.DataFrame)
        appraisals = dict(self.calculator(contents, **kwargs))
        results = pd.concat([contents] + list(appraisals.values()), axis=1)
        results = results.reset_index(drop=True, inplace=False)
        return results

    def calculator(self, contents, /, current, interest, **kwargs):
        assert isinstance(contents, pd.DataFrame)
        for appraisal, equation in self.equations.items():
            appraisals = equation(contents, current=current, interest=interest)
            assert isinstance(appraisals, pd.DataFrame)
            yield appraisal, appraisals

    @staticmethod
    def technicals(contents, technicals, /, **kwargs):
        assert isinstance(contents, pd.DataFrame) and isinstance(technicals, pd.DataFrame)
        mask = technicals["date"] == technicals["date"].max()
        technicals = technicals.where(mask).dropna(how="all", inplace=False)
        technicals = technicals.drop(columns="date", inplace=False)
        contents = contents.merge(technicals, how="left", on=list(Querys.Symbol), sort=False, suffixes=("", ""))
        return contents

    @property
    def equations(self): return self.__equations



