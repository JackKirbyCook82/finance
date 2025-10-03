# -*- coding: utf-8 -*-
"""
Created on Fri May 23 2025
@name:   Appraisal Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from abc import ABC, ABCMeta
from scipy.stats import norm

from finance.concepts import Concepts, Querys
from support.mixins import Emptying, Sizing, Partition, Logging
from support.meta import RegistryMeta
from calculations import Variables, Equations

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["AppraisalCalculator"]
__copyright__ = "Copyright 2025, Jack Kirby Cook"
__license__ = "MIT License"


class AppraisalEquationMeta(RegistryMeta, type(Equations.Vectorized.Table), ABCMeta): pass
class AppraisalEquation(Equations.Vectorized.Table, ABC, metaclass=AppraisalEquationMeta):
    τ = Variables.Dependent("τ", "tau", np.float32, function=lambda to, tτ: (np.datetime64(tτ, "ns") - np.datetime64(to, "ns")) / np.timedelta64(364, 'D'))

    tτ = Variables.Independent("tτ", "expire", np.datetime64, locator="expire")
    to = Variables.Constant("to", "current", np.datetime64, locator="current")

    i = Variables.Independent("i", "option", Concepts.Securities.Option, locator="option")
    j = Variables.Independent("j", "position", Concepts.Securities.Position, locator="position")
    x = Variables.Independent("x", "adjusted", np.float32, locator="adjusted")
    σ = Variables.Independent("σ", "volatility", np.float32, locator="volatility")
    μ = Variables.Independent("μ", "trend", np.float32, locator="trend")
    k = Variables.Independent("k", "strike", np.float32, locator="strike")
    r = Variables.Constant("r", "interest", np.float32, locator="interest")

    def execute(self, *args, **kwargs):
        yield self.τ()
        yield from super().execute(*args, **kwargs)


class BlackScholesEquation(AppraisalEquation, register=Concepts.Appraisal.BLACKSCHOLES):
    v = Variables.Dependent("v", "value", np.float32, function=lambda x, k, zx, zk, r, τ, i: x * norm.cdf(zx * int(i)) * int(i) - k * norm.cdf(zk * int(i)) * int(i) / np.exp(r * τ))

    zx = Variables.Dependent("zx", ("zscore", "itm"), np.float32, function=lambda zxk, zvt, zrt: zxk + zvt + zrt)
    zk = Variables.Dependent("zk", ("zscore", "otm"), np.float32, function=lambda zxk, zvt, zrt: zxk - zvt + zrt)

    zxk = Variables.Dependent("zxk", ("zscore", "strike"), np.float32, function=lambda x, k, σ, τ: np.log(x / k) / np.sqrt(τ) / σ)
    zvt = Variables.Dependent("zvt", ("zscore", "volatility"), np.float32, function=lambda σ, τ: np.sqrt(τ) * σ / 2)
    zrt = Variables.Dependent("zrt", ("zscore", "interest"), np.float32, function=lambda σ, r, τ: np.sqrt(τ) * r / σ)

    def execute(self, *args, **kwargs):
        yield self.v()
        yield from super().execute(*args, **kwargs)

class GreekEquation(AppraisalEquation, register=Concepts.Appraisal.GREEKS):
    Θ = Variables.Dependent("Θ", "theta", np.float32, function=lambda zx, zk, x, k, r, σ, τ, i: - norm.cdf(zk * int(i)) * int(i) * k * r / np.exp(r * τ) - norm.pdf(zx) * x * σ / np.sqrt(τ) / 2)
    P = Variables.Dependent("P", "rho", np.float32, function=lambda zk, k, r, τ, i: + norm.cdf(zk * int(i)) * int(i) * k * τ / np.exp(r * τ))
    Δ = Variables.Dependent("Δ", "delta", np.float32, function=lambda zx, i: + norm.cdf(zx * int(i)) * int(i))
    Γ = Variables.Dependent("Γ", "gamma", np.float32, function=lambda zx, x, σ, τ: + norm.pdf(zx) / np.sqrt(τ) / σ / x)
    V = Variables.Dependent("V", "vega", np.float32, function=lambda zx, x, τ: + norm.pdf(zx) * np.sqrt(τ) * x)

    def execute(self, *args, **kwargs):
        yield self.Θ()
        yield self.P()
        yield self.Δ()
        yield self.Γ()
        yield self.V()
        yield from super().execute(*args, **kwargs)


class AppraisalCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, appraisals, **kwargs):
        super().__init__(*args, **kwargs)
        equations = [equation for appraisal, equation in iter(AppraisalEquation) if appraisal in appraisals]
        self.__equation = AppraisalEquation + equations

    def execute(self, securities, *args, **kwargs):
        assert isinstance(securities, pd.DataFrame)
        if self.empty(securities): return
        querys = self.keys(securities, by=Querys.Settlement)
        querys = ",".join(list(map(str, querys)))
        securities = self.calculate(securities, *args, **kwargs)
        size = self.size(securities)
        self.console(f"{str(querys)}[{int(size):.0f}]")
        if self.empty(securities): return
        yield securities

    def calculate(self, securities, *args, current, interest, **kwargs):
        assert isinstance(securities, pd.DataFrame)
        parameters = dict(current=current, interest=interest)
        equation = self.equation(arguments=securities, parameters=parameters)
        results = equation(*args, **kwargs)
        assert isinstance(results, pd.DataFrame)
        securities = pd.concat([securities, results], axis=1)
        securities = securities.reset_index(drop=True, inplace=False)
        return securities

    @property
    def equation(self): return self.__equation



