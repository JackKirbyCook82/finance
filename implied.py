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
from enum import IntEnum
from numba import njit, prange
from abc import ABC, abstractmethod

from finance.concepts import Concepts, Querys
from calculations import Equation, Variables, Algorithms, Computations
from support.mixins import Emptying, Sizing, Partition, Logging, Naming

__author__ = "Jack Kirby Cook"
__all__ = ["ImpliedCalculator"]
__copyright__ = "Copyright 2025, Jack Kirby Cook"
__license__ = "MIT License"


@njit(fastmath=False, cache=True, inline="always")
def normcdf(z): return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

@njit(fastmath=True, cache=True, inline="always")
def normpdf(z): return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

@njit(fastmath=True, cache=True, inline="always")
def zitm(x, k, r, σ, τ): return math.log(x / k) / (σ * math.sqrt(τ)) + 0.5 * σ * math.sqrt(τ) + r * math.sqrt(τ) / σ

@njit(fastmath=True, cache=True, inline="always")
def zotm(x, k, r, σ, τ): return math.log(x / k) / (σ * math.sqrt(τ)) - 0.5 * σ * math.sqrt(τ) + r * math.sqrt(τ) / σ

@njit(fastmath=False, cache=True, inline="always")
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
    assert i == +1 or i == -1
    if i == +1: yl = max(0.0, x - k / math.exp(r * τ)); yh = x
    elif i == -1: yl = max(0.0, k / math.exp(r * τ) - x); yh = k / math.exp(r * τ)
    return yl, yh

@njit(fastmath=True, cache=True, inline="always")
def adaptive(x, k, τ):
    vmx = (3 * 3) + 1 * abs(math.log(x / k))
    τmx = max(τ, 1.0 / 365.0)
    σmx = min(math.sqrt(vmx / τmx), 20.0)
    return σmx

@njit(fastmath=True, cache=True, inline="always")
def brenner(mkt, x, k, τ, i):
    vτ = max(i * (x - k), 0.0)
    dy = max(mkt - vτ, 1e-12)
    x = max(x, 1e-12)
    σ = math.sqrt(2.0 * math.pi / τ) * (dy / x)
    σ = max(min(σ, 1), 0.05)
    return σ

@njit(fastmath=True, cache=True, inline="always")
def newton(mkt, x, k, r, σ, τ, i, /, low, high, tol, iters):
    for _ in range(iters):
        y = value(x, k, r, σ, τ, i)
        dy = y - mkt
        if abs(dy) < tol: return σ
        dσy = vega(x, k, r, σ, τ)
        if dσy <= 1e-12 or not math.isfinite(dσy): return math.nan
        dσ = dy / dσy
        if not math.isfinite(σ - dσ): return math.nan
        if not (low < σ - dσ < high): return math.nan
        if abs(dσ) <= 1.0: σ = σ - dσ
        else: σ = σ - math.copysign(1.0, dσ)
    return math.nan

@njit(fastmath=True, cache=True, inline="always")
def bisection(mkt, x, k, r, τ, i, /, low, high, tol):
    σl, σh, σmx = low, high, adaptive(x, k, τ)
    σh = min(σh, σmx)
    yl = value(x, k, r, σl, τ, i) - mkt
    yh = value(x, k, r, σh, τ, i) - mkt
    while yl * yh > 0.0 and σh < σmx:
        σh = 2.0 * σh
        σh = min(σh, σmx)
        yh = value(x, k, r, σh, τ, i) - mkt
    if yl * yh > 0.0: return math.nan
    for _ in range(100):
        σm = 0.5 * (σl + σh)
        ym = value(x, k, r, σm, τ, i) - mkt
        if abs(ym) < tol or (σh - σl) < 1e-12: return σm
        if yl * ym <= 0.0: σh = σm; yh = ym
        else: σl = σm; yl = ym
    return 0.5 * (σl + σh)

@njit(fastmath=True, cache=True, inline="always")
def volatility(mkt, x, k, r, τ, i, /, low, high, tol, iters):
    assert i == 1 or i == -1
    σ = brenner(mkt, x, k, τ, i)
    σ = newton(mkt, x, k, r, σ, τ, i, low=low, high=high, tol=tol, iters=iters)
    if not math.isnan(σ): return σ
    return bisection(mkt, x, k, r, τ, i, low=low, high=high, tol=tol)

@njit(fastmath=True, cache=True, inline="always")
def clipped(mkt, x, k, τ, yl, yh, /, low, high):
    vl, vh = mkt - yl, yh - mkt
    if vl <= vh: return low
    return min(high, adaptive(x, k, τ))

@njit(fastmath=True, cache=True, inline="always")
def fitted(mkt, x, k, r, τ, i, /, low, high, tol, iters):
    assert i == 1 or i == -1
    σl, σh = low, high
    for _ in range(iters):
        σml = σl + (σh - σl) / 3.0
        σmh = σh - (σh - σl) / 3.0
        yml = abs(value(x, k, r, σml, τ, i) - mkt)
        ymh = abs(value(x, k, r, σmh, τ, i) - mkt)
        if yml < ymh: σh = σmh
        else: σl = σml
        if (σh - σl) < tol: break
    return 0.5 * (σl + σh)

@njit(fastmath=True, cache=True, inline="always")
def residual(mkt, x, k, r, τ, i):
    assert i == 1 or i == -1
    yl, yh = boundary(x, k, r, τ, i)
    if math.isnan(yl) or math.isnan(yh): return math.nan
    if mkt < yl - 1e-12: return mkt - yl
    elif mkt > yh + 1e-12: return mkt - yh
    else: return 0.0

@njit(fastmath=True, cache=True)
def implied_strict(mkt, x, k, r, τ, i, /, low, high, tol, iters):
    if τ <= 0.0: return math.nan
    yl, yh = boundary(x, k, r, τ, i)
    if mkt < yl - 1e-12 or mkt > yh + 1e-12: σ = math.nan
    else: σ = volatility(mkt, x, k, r, τ, i, low=low, high=high, tol=tol, iters=iters)
    return σ

@njit(fastmath=True, cache=True)
def implied_clipped(mkt, x, k, r, τ, i, /, low, high, tol, iters):
    if τ <= 0.0: return math.nan
    yl, yh = boundary(x, k, r, τ, i)
    if mkt < yl - 1e-12: σ = low
    elif mkt > yh + 1e-12: σ = min(high, adaptive(x, k, τ))
    else: σ = volatility(mkt, x, k, r, τ, i, low=low, high=high, tol=tol, iters=iters)
    if math.isnan(σ): σ = clipped(mkt, x, k, τ, yl, yh, low=low, high=high)
    return σ

@njit(fastmath=True, cache=True)
def implied_fitted(mkt, x, k, r, τ, i, /, low, high, tol, iters):
    if τ <= 0.0: return math.nan
    yl, yh = boundary(x, k, r, τ, i)
    high = min(high, adaptive(x, k, τ))
    if mkt < yl - 1e-12 or mkt > yh + 1e-12: σ = fitted(mkt, x, k, r, τ, i, low=low, high=high, tol=tol, iters=iters)
    else: σ = volatility(mkt, x, k, r, τ, i, low=low, high=high, tol=tol, iters=iters)
    if math.isnan(σ): σ = fitted(mkt, x, k, r, τ, i, low=low, high=high, tol=tol, iters=iters)
    return σ

@njit(fastmath=True, cache=True)
def residual_strict(mkt, x, k, r, τ, i, /, low, high, tol, iters):
    if τ <= 0.0: return math.nan
    ε = residual(mkt, x, k, r, τ, i)
    return ε

@njit(fastmath=True, cache=True)
def residual_clipped(mkt, x, k, r, τ, i, /, low, high, tol, iters):
    if τ <= 0.0: return math.nan
    ε = residual(mkt, x, k, r, τ, i)
    return ε

@njit(fastmath=True, cache=True)
def residual_fitted(mkt, x, k, r, τ, i, /, low, high, tol, iters):
    if τ <= 0.0: return math.nan
    yl, yh = boundary(x, k, r, τ, i)
    high = min(high, adaptive(x, k, τ))
    if mkt < yl - 1e-12:
        σ = fitted(mkt, x, k, r, τ, i, low=low, high=high, tol=tol, iters=iters)
        ε = value(x, k, r, σ, τ, i) - mkt
    elif mkt > yh + 1e-12:
        σ = fitted(mkt, x, k, r, τ, i, low=low, high=high, tol=tol, iters=iters)
        ε = value(x, k, r, σ, τ, i) - mkt
    else: ε = residual(mkt, x, k, r, τ, i)
    return ε

@njit(fastmath=True, cache=True, parallel=True)
def volatility_calculation(mkt, x, k, r, τ, i, /, mode=0, low=1e-9, high=5.0, tol=1e-8, iters=12):
    shape = mkt.shape[0]
    σ = np.empty(shape, dtype=np.float64)
    if mode == 0:
        for idx in prange(shape):
            σ[idx] = implied_strict(mkt[idx], x[idx], k[idx], r[idx], τ[idx], i[idx], low=low, high=high, tol=tol, iters=iters)
    elif mode == 1:
        for idx in prange(shape):
            σ[idx] = implied_clipped(mkt[idx], x[idx], k[idx], r[idx], τ[idx], i[idx], low=low, high=high, tol=tol, iters=iters)
    elif mode == 2:
        for idx in prange(shape):
            σ[idx] = implied_fitted(mkt[idx], x[idx], k[idx], r[idx], τ[idx], i[idx], low=low, high=high, tol=tol, iters=iters)
    else:
        for idx in prange(shape): σ[idx] = math.nan
    return σ

@njit(fastmath=True, cache=True, parallel=True)
def residual_calculation(mkt, x, k, r, τ, i, /, mode=0, low=1e-9, high=5.0, tol=1e-8, iters=12):
    shape = mkt.shape[0]
    ε = np.empty(shape, dtype=np.float64)
    if mode == 0:
        for idx in prange(shape):
            ε[idx] = residual_strict(mkt[idx], x[idx], k[idx], r[idx], τ[idx], i[idx], low=low, high=high, tol=tol, iters=iters)
    elif mode == 1:
        for idx in prange(shape):
            ε[idx] = residual_clipped(mkt[idx], x[idx], k[idx], r[idx], τ[idx], i[idx], low=low, high=high, tol=tol, iters=iters)
    elif mode == 2:
        for idx in prange(shape):
            ε[idx] = residual_fitted(mkt[idx], x[idx], k[idx], r[idx], τ[idx], i[idx], low=low, high=high, tol=tol, iters=iters)
    else:
        for idx in prange(shape): ε[idx] = math.nan
    return ε


class ImpliedMode(IntEnum): STRICT, CLIPPED, FITTED = 0, 1, 2
class Calculation(Naming, ABC, fields=["mode", "low", "high", "tol", "iters"]):
    def __call__(self, mkt, x, k, r, τ, i):
        arguments = (mkt, x, k, r, τ.astype(np.float32), i.astype(np.int32))
        arguments = [argument.to_numpy() if isinstance(arguments, pd.Series) else np.array(argument) for argument in arguments]
        arguments = np.broadcast_arrays(*arguments)
        parameters = dict(mode=int(self.mode), low=self.low, high=self.high, tol=self.tol, iters=self.iters)
        results = self.execute(*arguments, **parameters)
        results = pd.Series(results)
        return results

    @abstractmethod
    def execute(self, *arguments, **parameters): pass

class ImpliedCalculation(Calculation):
    def execute(self, *arguments, **parameters): return volatility_calculation(*arguments, **parameters)

class ResidualCalculation(Calculation):
    def execute(self, *arguments, **parameters): return residual_calculation(*arguments, **parameters)


class ImpliedEquation(Computations.Table, Algorithms.UnVectorized.Table, Equation, ABC):
    λ = Variables.Dependent("λ", "implied", np.float32, function=ImpliedCalculation(mode=ImpliedMode.FITTED, low=1e-9, high=5.0, tol=1e-8, iters=12))
    ε = Variables.Dependent("ε", "residual", np.float32, function=ResidualCalculation(mode=ImpliedMode.FITTED, low=1e-9, high=5.0, tol=1e-8, iters=12))
    τ = Variables.Dependent("τ", "tau", np.int32, function=lambda to, tτ: (pd.to_datetime(tτ) - pd.to_datetime(to)).dt.days / 365)

    mkt = Variables.Independent("mkt", "price", np.float32, locator="price")
    tτ = Variables.Independent("tτ", "expire", np.datetime64, locator="expire")
    to = Variables.Constant("to", "current", np.datetime64, locator="current")

    x = Variables.Independent("x", "adjusted", np.float32, locator="adjusted")
    i = Variables.Independent("i", "option", Concepts.Securities.Option, locator="option")
    j = Variables.Independent("j", "position", Concepts.Securities.Position, locator="position")
    k = Variables.Independent("k", "strike", np.float32, locator="strike")
    r = Variables.Constant("r", "interest", np.float32, locator="interest")

    def execute(self, contents, /, current, interest):
        parameters = dict(current=current, interest=interest)
        yield from super().execute(contents, **parameters)
        yield self.λ(contents, **parameters)
        yield self.ε(contents, **parameters)


class ImpliedCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__equation = ImpliedEquation(*args, **kwargs)

    def execute(self, contents, /, **kwargs):
        assert isinstance(contents, (pd.DataFrame, types.NoneType))
        if self.empty(contents): return
        querys = self.keys(contents, by=Querys.Settlement)
        querys = ",".join(list(map(str, querys)))
        results = self.calculate(contents, **kwargs)
        size = self.size(results)
        self.console(f"{str(querys)}[{int(size):.0f}]")
        if self.empty(results): return
        yield results

    def calculate(self, contents, /, current, interest, **kwargs):
        assert isinstance(contents, pd.DataFrame)
        implications = self.equation(contents, current=current, interest=interest)
        assert isinstance(implications, pd.DataFrame)
        results = pd.concat([contents, implications], axis=1)
        results = results.reset_index(drop=True, inplace=False)
        return results

    @property
    def equation(self): return self.__equation


