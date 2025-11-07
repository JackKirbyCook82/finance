# -*- coding: utf-8 -*-
"""
Created on Fri May 23 2025
@name:   Appraisal Objects
@author: Jack Kirby Cook

"""

import math
import numpy as np
import pandas as pd
from abc import ABC
from numba import njit, prange

from finance.concepts import Concepts, Querys
from calculations import Equation, Variables, Algorithms, Computations
from support.mixins import Emptying, Sizing, Partition, Logging

__author__ = "Jack Kirby Cook"
__all__ = ["ImpliedCalculator"]
__copyright__ = "Copyright 2025, Jack Kirby Cook"
__license__ = "MIT License"


@njit(fastmath=True, cache=True, inline="always")
def normcdf(z): return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

@njit(fastmath=True, cache=True, inline="always")
def normpdf(z): return 1 / math.exp(0.5 * z * z) / math.sqrt(2.0 * math.pi)

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
    for _ in prange(iters):
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
    shape = mkt.shape[0]
    result = np.empty(shape, dtype=np.float32)
    for idx in prange(shape):
        result[idx] = implied(mkt[idx], x[idx], k[idx], r[idx], τ[idx], i[idx], low=low, high=high, tol=tol, iters=iters)
    return result


class ImpliedFunction(object):
    def __init__(self, /, low, high, tol, iters):
        self.low = low
        self.high = high
        self.tol = tol
        self.iters = iters

    def __call__(self, mkt, x, k, r, τ, i):
        arguments = (mkt, x, k, r, τ.astype(np.int32), i.astype(np.int32))
        arguments = [argument.to_numpy() if isinstance(arguments, pd.Series) else np.array(argument) for argument in arguments]
        arguments = np.broadcast_arrays(*arguments)
        parameters = dict(low=self.low, high=self.high, tol=self.tol, iters=self.iters)
        results = calculation(*arguments, **parameters)
        results = pd.Series(results)
        return results


class ImpliedEquation(Computations.Table, Algorithms.UnVectorized.Table, Equation, ABC):
    λ = Variables.Dependent("λ", "implied", np.float32, function=ImpliedFunction(low=1e-9, high=5.0, tol=1e-8, iters=12))
    τ = Variables.Dependent("τ", "tau", np.int32, function=lambda to, tτ: (pd.to_datetime(tτ) - pd.to_datetime(to)).dt.days)

    mkt = Variables.Independent("mkt", "price", np.float32, locator="price")
    tτ = Variables.Independent("tτ", "expire", np.datetime64, locator="expire")
    to = Variables.Constant("to", "current", np.datetime64, locator="current")

    x = Variables.Independent("x", "underlying", np.float32, locator="underlying")
    i = Variables.Independent("i", "option", Concepts.Securities.Option, locator="option")
    j = Variables.Independent("j", "position", Concepts.Securities.Position, locator="position")
    k = Variables.Independent("k", "strike", np.float32, locator="strike")
    r = Variables.Constant("r", "interest", np.float32, locator="interest")

    def execute(self, securities, /, current, interest):
        parameters = dict(current=current, interest=interest)
        yield from super().execute(securities, **parameters)
        yield self.λ(securities, **parameters)


class ImpliedCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__equation = ImpliedEquation(*args, **kwargs)

    def execute(self, securities, /, **kwargs):
        assert isinstance(securities, pd.DataFrame)
        if self.empty(securities): return
        querys = self.keys(securities, by=Querys.Settlement)
        querys = ",".join(list(map(str, querys)))
        securities = self.calculate(securities, **kwargs)
        size = self.size(securities)
        self.console(f"{str(querys)}[{int(size):.0f}]")
        if self.empty(securities): return
        yield securities

    def calculate(self, securities, *args, current, interest, **kwargs):
        assert isinstance(securities, pd.DataFrame)
        implications = self.equation(securities, current=current, interest=interest)
        assert isinstance(implications, pd.DataFrame)
        results = pd.concat([securities, implications], axis=1)
        results = results.reset_index(drop=True, inplace=False)
        return results

    @property
    def equation(self): return self.__equation


