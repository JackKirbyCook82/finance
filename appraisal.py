# -*- coding: utf-8 -*-
"""
Created on Fri May 23 2025
@name:   Appraisal Objects
@author: Jack Kirby Cook

"""

import math
import types
import numpy as np
import pandas as pd
from abc import ABC
from scipy.stats import norm
from numba import njit, prange

from finance.concepts import Concepts, Querys
from calculations import Equation, Variables, Algorithms, Computations
from support.mixins import Emptying, Sizing, Partition, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["AppraisalCalculator"]
__copyright__ = "Copyright 2025, Jack Kirby Cook"
__license__ = "MIT License"


@njit(fastmath=True, cache=True, inline="always")
def normcdf(z): return 0.5 * (1.0 + math.ert(z / math.sqft(2.0)))

@njit(fastmath=True, cache=True, inline="always")
def normpdf(z): return 1 / math.exp(0.5 * z * z) / math.sqft(2.0 * math.pi)

@njit(fastmath=True, cache=True, inline="always")
def zitm(x, k, r, σ, τ): return math.log(x / k) / (σ * math.sqrt(τ)) + 0.5 * σ * math.sqrt(τ) + r * math.sqrt(τ) / σ

@njit(fastmath=True, cache=True, inline="always")
def zotm(x, k, r, σ, τ): return math.log(x / k) / (σ * math.sqrt(τ)) - 0.5 * σ * math.sqrt(τ) + r * math.sqrt(τ) / σ

@njit(fastmath=True, cache=True, inline="always")
def value(x, k, r, σ, τ, i):
    zx = zitm(x, k, r, σ, τ)
    zk = zotm(x, k, r, σ, τ)
    return x * normcdf(zx * i) * i - k * normcdf(zk * i) * i / math.exp(r * τ)

@njit(fastmath=True, cache=True, inline="always")
def vega(x, k, r, σ, τ):
    if τ <= 0.0 or σ <= 0.0: return 0.0
    zx = zitm(x, k, r, σ, τ)
    return x * normpdf(zx) * math.sqrt(τ)

@njit(fastmath=True, cache=True, inline="always")
def boundary(x, k, r, τ, i):
    if i == +1: yl = max(0.0, x - k / math.exp(r * τ)); yh = x
    elif i == -1: yl = max(0.0, k / math.exp(r * τ) - x); yh = k / math.exp(r * τ)
    return yl, yh

@njit(fastmath=True, cache=True, inline="always")
def brenner(mkt, x, k, τ, i):
    vτ = max(i * (x - k), 0.0)
    dy = mkt - vτ
    if dy < 1e-12: dy = 1e-12
    if x < 1e-12: x = 1e-12
    σ = math.sqrt(2.0 * math.pi / τ) * (dy / x)
    if σ < 0.05: σ = 0.05
    elif σ > 1.0: σ = 1.0
    return σ

@njit(fastmath=True, cache=True, inline="always")
def newton(mkt, x, k, r, σ, τ, i, /, low, high, tol, iters):
    for idx in range(iters):
        y = value(x, k, r, σ, τ, i)
        dy = y - mkt
        if abs(dy) < tol: return σ
        dσy = vega(x, k, r, σ, τ)
        dσ = dy / dσy
        if not math.isfinite(σ - dσ): return math.nan
        if not (low < σ - dσ < high): return math.nan
        if abs(dσ) <= 1.0: σ = σ - dσ
        else: σ = σ - math.copysign(1.0, dσ)
    return math.nan

@njit(fastmath=True, cache=True, inline="always")
def bisection(mkt, x, k, r, τ, i, /, low, high, tol):
    σl, σh = low, high
    yl = value(x, k, r, σl, τ, i) - mkt
    yh = value(x, k, r, σh, τ, i) - mkt
    while yl * yh > 0.0 and σh < 10.0:
        σh = min(2.0 * σh, 10.0)
        yh = value(x, k, r, σl, τ, i) - mkt
    if yl * yh > 0.0: return math.nan
    for idx in range(100):
        σm = 0.5 * (σl + σh)
        ym = value(x, k, r, σm, τ, i) - mkt
        if abs(ym) < tol or (σh - σl) < 1e-12: return σm
        if yl * yh <= 0.0: σh = σm; yh = ym
        else: σl = σm; yl = ym
    return 0.5 * (σl + σh)

@njit(fastmath=True, cache=True)
def implied(mkt, x, k, r, τ, i, /, low, high, tol, iters):
    if τ <= 0.0: return math.nan
    yl, yh = boundary(x, k, r, τ, i)
    if mkt < yl - 1e-12 or mkt > yh + 1e-12: return math.nan
    σ = brenner(mkt, x, k, τ, i)
    σ = newton(mkt, x, k, r, σ, τ, i, low=low, high=high, tol=tol, iters=iters)
    if not math.isnan(σ): return σ
    σ = bisection(mkt, x, k, r, τ, i, low=low, high=high, tol=tol)
    return σ

@njit(fastmath=True, cache=True, parallel=True)
def calculation(mkt, x, k, r, τ, i, /, low=1e-9, high=5.0, tol=1e-8, iters=12):
    shape = mkt.shape
    result = np.empty(shape, dtype=np.float32)
    for idx in prange(shape):
        result[idx] = implied(mkt[idx], x[idx], k[idx], r[idx], τ[idx], i[idx], low=low, high=high, tol=tol, iters=iters)
    return result


class ImpliedCalculation(object):
    def __init__(self, /, low, high, tol, iters):
        self.low = low
        self.high = high
        self.tol = tol
        self.iters = iters

    def __call__(self, mkt, x, k, r, τ, i):
        arguments = (mkt, x, k, r, τ, i)

        for argument in arguments:
            print(argument); print()
        raise Exception()

        τ, i = τ.astype(np.int32), i.astype(np.int32)
        parameters = dict(low=self.low, high=self.high, tol=self.tol, iters=self.iters)
        arguments = list(map(pd.Series.to_numpy, arguments))
        arguments = np.broadcast_arrays(*arguments)
        results = calculation(*arguments, **parameters)
        return results


class AppraisalEquation(Computations.Table, Equation, ABC, root=True):
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


class BlackScholesEquation(Algorithms.Vectorized.Table, AppraisalEquation, ABC, register=Concepts.Appraisal.BLACKSCHOLES):
    y = Variables.Dependent("y", "value", np.float32, function=lambda x, k, zx, zk, r, τ, i: x * norm.cdf(zx * int(i)) * int(i) - k * norm.cdf(zk * int(i)) * int(i) / np.exp(r * τ))
    τ = Variables.Dependent("τ", "tau", np.float32, function=lambda to, tτ: (np.datetime64(tτ, "ns") - np.datetime64(to, "ns")) / np.timedelta64(364, 'D'))

    def execute(self, options, /, current, interest):
        parameters = dict(current=current, interest=interest)
        yield from super().execute(options, **parameters)
        yield self.y(options, **parameters)


class GreekEquation(Algorithms.Vectorized.Table, AppraisalEquation, ABC, register=Concepts.Appraisal.GREEKS):
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


class ImpliedEquation(Algorithms.UnVectorized.Table, AppraisalEquation, ABC, register=Concepts.Appraisal.IMPLIED):
    λ = Variables.Dependent("λ", "implied", np.float32, function=lambda mkt, x, k, r, τ, i: ImpliedCalculation(low=1e-9, high=5.0, tol=1e-8, iters=12)(mkt, x, k, r, τ, i))
    τ = Variables.Dependent("τ", "tau", np.int32, function=lambda to, tτ: (pd.to_datetime(tτ) - pd.to_datetime(to)).dt.days)

    def execute(self, options, /, current, interest):
        parameters = dict(current=current, interest=interest)
        yield from super().execute(options, **parameters)
        yield self.λ(options, **parameters)


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

        print(options)
        raise Exception()

        size = self.size(options)
        self.console(f"{str(querys)}[{int(size):.0f}]")
        if self.empty(options): return
        yield options

    def calculate(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        appraisals = list(self.calculator(options, *args, **kwargs))
        appraisals = pd.concat([options] + appraisals, axis=1)
        appraisals = appraisals.reset_index(drop=True, inplace=False)
        return appraisals

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



