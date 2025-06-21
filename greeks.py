# -*- coding: utf-8 -*-
"""
Created on Fri May 23 2025
@name:   Greeks Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from abc import ABC
from scipy.stats import norm

from finance.variables import Variables, Querys
from support.calculations import Calculation, Equation, Variable
from support.mixins import Emptying, Sizing, Partition, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["GreekCalculator"]
__copyright__ = "Copyright 2025, Jack Kirby Cook"
__license__ = "MIT License"


class GreekEquation(Equation, ABC, datatype=pd.Series, vectorize=True):
    τ = Variable.Dependent("τ", "tau", np.float32, function=lambda to, tτ: (np.datetime64(tτ, "ns") - np.datetime64(to, "ns")) / np.timedelta64(364, 'D'))
    Δ = Variable.Dependent("Δ", "delta", np.float32, function=lambda zx, i: + norm.cdf(zx * int(i)) * int(i))
    Γ = Variable.Dependent("Γ", "gamma", np.float32, function=lambda zx, x, σ, τ: + norm.pdf(zx) / np.sqrt(τ) / σ / x)
    P = Variable.Dependent("P", "rho", np.float32, function=lambda zk, k, r, τ, i: + norm.cdf(zk * int(i)) * int(i) * k * τ / np.exp(r * τ))
    V = Variable.Dependent("V", "vega", np.float32, function=lambda zx, x, τ: + norm.pdf(zx) * np.sqrt(τ) * x)
    Θ = Variable.Dependent("Θ", "theta", np.float32, function=lambda zx, zk, x, k, r, σ, τ, i: - norm.cdf(zk * int(i)) * int(i) * k * r / np.exp(r * τ) - norm.pdf(zx) * x * σ / np.sqrt(τ) / 2)

    v = Variable.Dependent("v", "value", np.float32, function=lambda vx, vk, i: (vx - vk) * int(i))
    vx = Variable.Dependent("yx", "underlying", np.float32, function=lambda zx, x, τ, q, i: norm.cdf(zx * int(i)) * x)
    vk = Variable.Dependent("yk", "strike", np.float32, function=lambda zk, k, τ, r, i: norm.cdf(zk * int(i)) * k / np.exp(r * τ))

    zx = Variable.Dependent("zx", "underlying", np.float32, function=lambda zxk, zvt, zrt: zxk + zvt + zrt)
    zk = Variable.Dependent("zx", "strike", np.float32, function=lambda zxk, zvt, zrt: zxk - zvt + zrt)

    zxk = Variable.Dependent("zxk", "strike", np.float32, function=lambda x, k, σ, τ: np.log(x / k) / np.sqrt(τ) / σ)
    zvt = Variable.Dependent("zvt", "volatility", np.float32, function=lambda σ, τ: np.sqrt(τ) * σ / 2)
    zrt = Variable.Dependent("zrt", "interest", np.float32, function=lambda σ, r, τ: np.sqrt(τ) * r / σ)

    tτ = Variable.Independent("tτ", "expire", np.datetime64, locator="expire")
    to = Variable.Constant("to", "current", np.datetime64, locator="current")

    x = Variable.Independent("x", "adjusted", np.float32, locator="adjusted")
    σ = Variable.Independent("σ", "volatility", np.float32, locator="volatility")
    μ = Variable.Independent("μ", "trend", np.float32, locator="trend")
    i = Variable.Independent("i", "option", Variables.Securities.Option, locator="option")
    k = Variable.Independent("k", "strike", np.float32, locator="strike")
    r = Variable.Constant("r", "interest", np.float32, locator="interest")
    q = Variable.Constant("q", "dividend", np.float32, locator="dividend")

    def execute(self, *args, **kwargs):
        yield from super().execute(*args, **kwargs)
        yield self.v()
        yield self.Δ()
        yield self.Γ()
        yield self.Θ()
        yield self.P()
        yield self.V()


class GreekCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__calculation = Calculation[pd.Series](*args, required=GreekEquation, **kwargs)

    def execute(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if self.empty(options): return
        querys = self.keys(options, by=Querys.Settlement)
        querys = ",".join(list(map(str, querys)))
        options = self.calculate(options, *args, **kwargs)
        size = self.size(options)
        self.console(f"{str(querys)}[{int(size):.0f}]")
        if self.empty(options): return
        yield options

    def calculate(self, options, *args, current, interest, dividend, **kwargs):
        assert isinstance(options, pd.DataFrame)
        parameters = dict(current=current, interest=interest, dividend=dividend)
        greeks = self.calculation(options, *args, **parameters, **kwargs)
        assert isinstance(greeks, pd.DataFrame)
        options = pd.concat([options, greeks], axis=1)
        options = options.reset_index(drop=True, inplace=False)
        return options

    @property
    def calculation(self): return self.__calculation



