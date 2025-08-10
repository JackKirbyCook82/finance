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
from support.equations import Calculation, Equation, Variable
from support.mixins import Emptying, Sizing, Partition, Logging
from support.calculus import Integral, Function

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["GreekCalculator"]
__copyright__ = "Copyright 2025, Jack Kirby Cook"
__license__ = "MIT License"


class ValueEquation(Equation, ABC, datatype=pd.Series, vectorize=True):
    τ = Variable.Dependent("τ", "tau", np.float32, function=lambda to, tτ: (np.datetime64(tτ, "ns") - np.datetime64(to, "ns")) / np.timedelta64(364, 'D'))

    tτ = Variable.Independent("tτ", "expire", np.datetime64, locator="expire")
    to = Variable.Constant("to", "current", np.datetime64, locator="current")

    i = Variable.Independent("i", "option", Variables.Securities.Option, locator="option")
    j = Variable.Independent("j", "position", Variables.Securities.Position, locator="position")
    x = Variable.Independent("x", "adjusted", np.float32, locator="adjusted")
    σ = Variable.Independent("σ", "volatility", np.float32, locator="volatility")
    μ = Variable.Independent("μ", "trend", np.float32, locator="trend")
    k = Variable.Independent("k", "strike", np.float32, locator="strike")
    r = Variable.Constant("r", "interest", np.float32, locator="interest")

class BlackScholesEquation(ValueEquation):
    zx = Variable.Dependent("zx", "underlying", np.float32, function=lambda zxk, zvt, zrt: zxk + zvt + zrt)
    zk = Variable.Dependent("zx", "strike", np.float32, function=lambda zxk, zvt, zrt: zxk - zvt + zrt)

    zxk = Variable.Dependent("zxk", "strike", np.float32, function=lambda x, k, σ, τ: np.log(x / k) / np.sqrt(τ) / σ)
    zvt = Variable.Dependent("zvt", "volatility", np.float32, function=lambda σ, τ: np.sqrt(τ) * σ / 2)
    zrt = Variable.Dependent("zrt", "interest", np.float32, function=lambda σ, r, τ: np.sqrt(τ) * r / σ)

class BlackScholesCallEquation(BlackScholesEquation):
    v = Variable.Dependent("v", "value", np.float32, function=lambda x, k, zx, zk, r, τ: x * norm.cdf(+zx) - k * norm.cdf(+zk) / np.exp(r * τ))

class BlackScholesPutEquation(BlackScholesEquation):
    v = Variable.Dependent("v", "value", np.float32, function=lambda x, k, zx, zk, r, τ: k * norm.cdf(-zk) / np.exp(r * τ) - x * norm.cdf(-zx))

class GreekEquation(Equation, ABC, datatype=pd.Series, vectorize=True):
    Θ = Variable.Dependent("Θ", "theta", np.float32, function=lambda zx, zk, x, k, r, σ, τ, i: - norm.cdf(zk * int(i)) * int(i) * k * r / np.exp(r * τ) - norm.pdf(zx) * x * σ / np.sqrt(τ) / 2)
    P = Variable.Dependent("P", "rho", np.float32, function=lambda zk, k, r, τ, i: + norm.cdf(zk * int(i)) * int(i) * k * τ / np.exp(r * τ))
    Δ = Variable.Dependent("Δ", "delta", np.float32, function=lambda zx, i: + norm.cdf(zx * int(i)) * int(i))
    Γ = Variable.Dependent("Γ", "gamma", np.float32, function=lambda zx, x, σ, τ: + norm.pdf(zx) / np.sqrt(τ) / σ / x)
    V = Variable.Dependent("V", "vega", np.float32, function=lambda zx, x, τ: + norm.pdf(zx) * np.sqrt(τ) * x)


class AnsatzFunction(Function, signature="(α,β,γ,ξ,τ,r)->ϕ(σ,u,w,x)"):
    a = lambda α, β: α * β
    b = lambda α, γ, ξ, u: α - ξ * γ * u * 1j
    c = lambda τ, b, d, f, g: (b - d) * τ - 2 * np.log(f / (1 - g))
    d = lambda γ, b, u: np.sqrt(b ** 2 + γ ** 2 * (u ** 2 + u * 1j))
    e = lambda τ, d: 1 - 1 / np.exp(d * τ)
    f = lambda τ, d, g: 1 - g / np.exp(d * τ)
    g = lambda b, d: (b - d) / (b + d)

    A = lambda γ, τ, b, d, e, f, g: ((b - d) / γ**2) * ((b - d) * τ - 2 * np.log(f / (1 - g)))
    B = lambda γ, τ, a, c, r, u, w: (u * 1j - w) * r * τ + (a / γ**2) * c
    ϕ = lambda σ, A, B, u, w, x: np.exp(A + B * σ + (u * 1j - w) * x)

class RiccatiIntegral(Integral, bounds=(1e-8, 200), signature="(α,β,γ,ξ,τ,k,r)->Ψ(σ,w,x)"):
    Ψ = lambda ϕ, k: lambda u: ϕ / (np.exp(u * 1j * np.log(k)) * u * 1j)
    ϕ = lambda α, β, γ, ξ, σ, τ, r, w, x: lambda u: AnsatzFunction(α, β, γ, ξ, τ, r)(σ, u, w, x)

class RiccatiFunction(Function, signature=""):
    Σx = lambda α, β, γ, ξ, σ, τ, k, r: RiccatiIntegral(α, β, γ, ξ, τ, k, r)(σ, w, x)
    Σk = lambda α, β, γ, ξ, σ, τ, k, r: RiccatiIntegral(α, β, γ, ξ, τ, k, r)(σ, w, x)


# class RiccatiIntegral(Integral, parameters="α β γ σ ξ τ x r", domain="u", range="Σ"):
#     R = lambda ϕ, k: lambda u: RiccatiFunction(ϕ, k)(u)
# class RiccatiStrikeIntegral(RiccatiIntegral):
#     ϕ = lambda α, β, γ, σ, ξ, τ, x, r: lambda u: AnsatzFunction(α, β, γ, ξ, τ, r)(u - 0, np.log(x), u, 1)
# class RiccatiStockIntegral(RiccatiIntegral):
#     ϕ = lambda α, β, γ, σ, ξ, τ, x, r: lambda u: AnsatzFunction(α, β, γ, ξ, τ, r)(u - 1, np.log(x), u, 0)
# class RiccatiFunction(Function):
#    Rx = lambda Σx: 0.5 + (1 / np.pi) * Σx
#    Rk = lambda Σk: 0.5 + (1 / np.pi) * Σk
#    Σx = lambda : RiccatiIntegral()()
#    Σk = lambda : RiccatiIntegral()()
#    Σx = lambda : lambda : np.real(ϕx / (np.exp(np.log(k) * u * 1j) * u * 1j))
#    Σk = lambda : lambda : np.real(ϕk / (np.exp(np.log(k) * u * 1j) * u * 1j))
#    ψ = lambda ϕ, k: lambda u: np.real(ϕ / (np.exp(np.log(k) * u * 1j) * u * 1j))
# class UnderlyingRiccatiIntegral(RiccatiIntegral):
#     ϕx = lambda x, σ, τ, r, C, D: lambda u: ((u - 0) * 1j - 1) * r * τ + C + D * σ + ((u - 0) * 1j - 1) * np.log(x)
#     Σx = lambda ϕx, k: lambda u: np.real(ϕx / (np.exp(np.log(k) * u * 1j) * u * 1j))
#    ϕk = lambda x, σ, A, B: lambda u: np.exp(A, B * σ + (u * 1j - XXX) * np.log(x))
#    R = lambda τ, r: lambda u: (u * 1j - 0) * r * τ
# class StrikeRiccatiIntegral(RiccatiIntegral):
#     b = lambda α, γ, ξ: lambda u: α - ξ * γ * (u ) * 1j
#     ϕk = lambda x, σ, τ, r, C, D: lambda u: ((u - 1) * 1j - 0) * r * τ + C + D * σ + ((u - 1) * 1j - 0) * np.log(x)
#     Σk = lambda ϕk, k: lambda u: np.real(ϕk / (np.exp(np.log(k) * u * 1j) * u * 1j))
#    ϕx = lambda x, σ, A, B: lambda u: np.exp(A, B * σ + (u * 1j - XXX) * np.log(x))
#    R = lambda τ, r: lambda u: (u * 1j - 1) * r * τ
# class HestonEquation(ValueEquation):
#     zx = Variable.Dependent("zx", "underlying", np.float32, function=lambda Σx: 0.5 + 1 / np.pi * Σx)
#     zk = Variable.Dependent("zk", "strike", np.float32, function=lambda Σk: 0.5 + 1 / np.pi * Σk)
#     Σx = Variable.Dependent("Σx", "underlying", np.float32, function=lambda x, k, τ, r, λ, α, β, γ: RiccatiIntegral(1e-8, 200).Σx(x, k, τ, r, λ, α, β, γ))
#     Σk = Variable.Dependent("Σk", "strike", np.float32, function=lambda x, k, τ, r, λ, α, β, γ: RiccatiIntegral(1e-8, 200).Σk(x, k, τ, r, λ, α, β, γ))
#    Σx = Variable.Dependent("Σx", "underlying", np.float32, function=lambda ?x: np.quad(?x, 1e-8, 200, limit=200)[0])
#    Σk = Variable.Dependent("Σk", "strike", np.float32, function=lambda ?k: np.quad(?k, 1e-8, 200, limit=200)[0])
#    fx = Variable.Dependent("fx", "underlying", types.LambdaType, function=lambda : lambda : )
#    fk = Variable.Dependent("fk", "strike", types.LambdaType, function=lambda : lambda : )
# class HestonCallEquation(HestonEquation):
#     v = Variable.Dependent("v", "value", np.float32, function=lambda x, k, zx, zk, r, τ: x * zx - k * zk / np.exp(r * τ))
# class HestonPutEquation(HestonEquation):
#     v = Variable.Dependent("v", "value", np.float32, function=lambda x, k, zx, zk, r, τ: k * (1 - zk) / np.exp(r * τ) - x * (1 - zx))


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



