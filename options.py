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

from finance.concepts import Concepts, Querys
from support.mixins import Emptying, Sizing, Partition, Logging
from calculations import Variables, Equations

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["GreekCalculator"]
__copyright__ = "Copyright 2025, Jack Kirby Cook"
__license__ = "MIT License"


# class AnsatzFunction(calc.Equations.Numeric, signature="(α,β,γ,ξ,τ,r,w)->ϕ(σ,u,x)"):
#     a = lambda α, β: α * β
#     b = lambda α, γ, ξ, u, w: α - ξ * γ * (u - w) * 1j
#     c = lambda τ, b, d, f, g: (b - d) * τ - 2 * np.log(f / (1 - g))
#     d = lambda γ, b, u, w: np.sqrt(b ** 2 + γ ** 2 * ((u - w) ** 2 + (u - w) * 1j))
#     e = lambda τ, d: 1 - 1 / np.exp(d * τ)
#     f = lambda τ, d, g: 1 - g / np.exp(d * τ)
#     g = lambda b, d: (b - d) / (b + d)
#
#     A = lambda γ, τ, b, d, e, f, g: ((b - d) / γ**2) * ((b - d) * τ - 2 * np.log(f / (1 - g)))
#     B = lambda γ, τ, a, c, r, u, w: ((u - w) * 1j - int(not w)) * r * τ + (a / γ**2) * c
#     ϕ = lambda σ, A, B, u, w, x: np.exp(A + B * σ + ((u - w) * 1j - int(not w)) * x)
#
# class RiccatiIntegral(calc.Equations.Integral, bounds=(1e-8, 200), signature="(α,β,γ,ξ,τ,k,r,w)->Ψ(σ,x)"):
#     Ψ = lambda ϕ, k: lambda u: ϕ / (np.exp(u * 1j * np.log(k)) * u * 1j)
#     ϕ = lambda α, β, γ, ξ, σ, τ, r, w, x: lambda u: AnsatzFunction(α, β, γ, ξ, τ, r, w)(σ, u, x)
#
# class RiccatiFunction(calc.Equations.Numeric, signature=""):
#     Σx = lambda α, β, γ, ξ, σ, τ, k, r, x: RiccatiIntegral(α, β, γ, ξ, τ, 0, k, r)(σ, x)
#     Σk = lambda α, β, γ, ξ, σ, τ, k, r, x: RiccatiIntegral(α, β, γ, ξ, τ, 1, k, r)(σ, x)
#     X = lambda Σx: 0.5 + (1 / np.pi) * Σx
#     K = lambda Σk: 0.5 + (1 / np.pi) * Σk


class ValueEquation(Equations.Table, ABC):
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

class BlackScholesEquation(ValueEquation):
    zx = Variables.Dependent("zx", ("zscore", "itm"), np.float32, function=lambda zxk, zvt, zrt: zxk + zvt + zrt)
    zk = Variables.Dependent("zx", ("zscore", "otm"), np.float32, function=lambda zxk, zvt, zrt: zxk - zvt + zrt)

    zxk = Variables.Dependent("zxk", ("zscore", "strike"), np.float32, function=lambda x, k, σ, τ: np.log(x / k) / np.sqrt(τ) / σ)
    zvt = Variables.Dependent("zvt", ("zscore", "volatility"), np.float32, function=lambda σ, τ: np.sqrt(τ) * σ / 2)
    zrt = Variables.Dependent("zrt", ("zscore", "interest"), np.float32, function=lambda σ, r, τ: np.sqrt(τ) * r / σ)

class BlackScholesCallEquation(BlackScholesEquation):
    v = Variables.Dependent("v", "value", np.float32, function=lambda x, k, zx, zk, r, τ: x * norm.cdf(+zx) - k * norm.cdf(+zk) / np.exp(r * τ))

class BlackScholesPutEquation(BlackScholesEquation):
    v = Variables.Dependent("v", "value", np.float32, function=lambda x, k, zx, zk, r, τ: k * norm.cdf(-zk) / np.exp(r * τ) - x * norm.cdf(-zx))

class HestonCallEquation(ValueEquation):
    v = Variables.Dependent("v", "value", np.float32, function=lambda Σk, Σx, k, x, r, τ: x * Σx - k * Σk / np.exp(r * τ))

class HestonPutEquation(ValueEquation):
    v = Variables.Dependent("v", "value", np.float32, function=lambda Σk, Σx, k, x, r, τ: x * (Σx - 1) - k * (Σk - 1) / np.exp(r * τ))

class GreekEquation(Equations.Table, ABC):
    Θ = Variables.Dependent("Θ", "theta", np.float32, function=lambda zx, zk, x, k, r, σ, τ, i: - norm.cdf(zk * int(i)) * int(i) * k * r / np.exp(r * τ) - norm.pdf(zx) * x * σ / np.sqrt(τ) / 2)
    P = Variables.Dependent("P", "rho", np.float32, function=lambda zk, k, r, τ, i: + norm.cdf(zk * int(i)) * int(i) * k * τ / np.exp(r * τ))
    Δ = Variables.Dependent("Δ", "delta", np.float32, function=lambda zx, i: + norm.cdf(zx * int(i)) * int(i))
    Γ = Variables.Dependent("Γ", "gamma", np.float32, function=lambda zx, x, σ, τ: + norm.pdf(zx) / np.sqrt(τ) / σ / x)
    V = Variables.Dependent("V", "vega", np.float32, function=lambda zx, x, τ: + norm.pdf(zx) * np.sqrt(τ) * x)


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



