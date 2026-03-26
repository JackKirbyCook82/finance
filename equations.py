# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 2026
@name:   Finance Equations
@author: Jack Kirby Cook

"""

import math
from numba import njit

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Equations"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@njit(cache=True, inline="always")
def normcdf(z): return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

@njit(cache=True, inline="always")
def normpdf(z): return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

@njit(cache=True, inline="always")
def discount(r, τ): return math.exp(-r * τ)

@njit(cache=True, inline="always")
def error(y, x, k, τ, σ, i, r): return value(x, k, τ, σ, i, r) - y

@njit(cache=True, inline="always")
def intrinsic(x, k, τ, i, r):
    dcf = discount(r, τ)
    if i == +1: yτ = max(0.0, x - k * dcf)
    elif i == -1: yτ = max(0.0, k * dcf - x)
    return yτ

@njit(cache=True, inline="always")
def boundary(x, k, τ, i, r):
    assert i == +1 or i == -1
    dcf = discount(r, τ)
    if i == +1: yl = max(0.0, x - k * dcf); yh = x
    elif i == -1: yl = max(0.0, k * dcf - x); yh = k * dcf
    return yl, yh

@njit(cache=True, inline="always")
def valid(x, k, τ, i):
    positive = (x > 0.0 and k > 0.0 and τ > 0.0)
    option = (i == 1 or i == -1)
    finite = (math.isfinite(x) and math.isfinite(k) and math.isfinite(τ))
    return positive and option and finite

@njit(cache=True, inline="always")
def zitm(x, k, τ, σ, r):
    if x <= 0.0 or k <= 0.0 or σ <= 0.0 or τ <= 0.0: return math.nan
    fτσ = σ * math.sqrt(τ)
    return (math.log(x / k) + (r + 0.5 * σ * σ) * τ) / fτσ

@njit(cache=True, inline="always")
def zotm(x, k, τ, σ, r):
    if x <= 0.0 or k <= 0.0 or σ <= 0.0 or τ <= 0.0: return math.nan
    fτσ = σ * math.sqrt(τ)
    return zitm(x, k, τ, σ, r) - fτσ

@njit(cache=True, inline="always")
def value(x, k, τ, σ, i, r):
    if not valid(x, k, τ, i) or σ <= 0.0 or not math.isfinite(σ): return math.nan
    zx = zitm(x, k, τ, σ, r)
    zk = zx - σ * math.sqrt(τ)
    dcf = discount(r, τ)
    return i * (x * normcdf(i * zx) - k * dcf * normcdf(i * zk))

@njit(cache=True, inline="always")
def delta(x, k, τ, σ, i, r):
    """dy/dx"""
    if not valid(x, k, τ, i) or σ <= 0.0: return math.nan
    zx = zitm(x, k, τ, σ, r)
    return i * normcdf(i * zx)

@njit(cache=True, inline="always")
def gamma(x, k, τ, σ, i, r):
    """d²y/dx²"""
    if x <= 0.0 or k <= 0.0 or τ <= 0.0 or σ <= 0.0: return math.nan
    zx = zitm(x, k, τ, σ, r)
    return normpdf(zx) / (x * σ * math.sqrt(τ))

@njit(cache=True, inline="always")
def theta(x, k, τ, σ, i, r):
    """dy/dτ"""
    if not valid(x, k, τ, i) or σ <= 0.0: return math.nan
    zx = zitm(x, k, τ, σ, r)
    zk = zx - σ * math.sqrt(τ)
    dcf = discount(r, τ)
    fx = -x * normpdf(zx) * σ / (2.0 * math.sqrt(τ))
    fk = -i * r * k * dcf * normcdf(i * zk)
    return fx + fk

@njit(cache=True, inline="always")
def rho(x, k, τ, σ, i, r):
    """dy/dr"""
    if not valid(x, k, τ, i) or σ <= 0.0: return math.nan
    zk = zotm(x, k, τ, σ, r)
    dcf = discount(r, τ)
    return i * k * τ * dcf * normcdf(i * zk)

@njit(cache=True, inline="always")
def vega(x, k, τ, σ, i, r):
    """dy/dσ"""
    if x <= 0.0 or k <= 0.0 or τ <= 0.0 or σ <= 0.0: return 0.0
    zx = zitm(x, k, τ, σ, r)
    return x * normpdf(zx) * math.sqrt(τ)

@njit(cache=True, inline="always")
def vomma(x, k, τ, σ, i, r):
    """d²y/dσ²"""
    if x <= 0.0 or k <= 0.0 or τ <= 0.0 or σ <= 0.0: return math.nan
    zx = zitm(x, k, τ, σ, r)
    zk = zx - σ * math.sqrt(τ)
    dydσ = vega(x, k, τ, σ, i, r)
    return dydσ * zx * zk / max(σ, 1e-12)

@njit(cache=True, inline="always")
def vanna(x, k, τ, σ, i, r):
    """d²y/dx*dσ"""
    if x <= 0.0 or k <= 0.0 or τ <= 0.0 or σ <= 0.0: return math.nan
    zx = zitm(x, k, τ, σ, r)
    zk = zx - σ * math.sqrt(τ)
    return -normpdf(zx) * zk / max(σ, 1e-12)

@njit(cache=True, inline="always")
def charm(x, k, τ, σ, i, r):
    """d²y/dx*dτ"""
    if not valid(x, k, τ, i) or σ <= 0.0: return math.nan
    zx = zitm(x, k, τ, σ, r)
    zk = zx - σ * math.sqrt(τ)
    fk = 2.0 * r * τ - zk * σ * math.sqrt(τ)
    fτσ = 2.0 * τ * σ * math.sqrt(τ)
    return -i * normpdf(zx) * fk / fτσ

@njit(cache=True, inline="always")
def adaptive(x, k, τ):
    xk = abs(math.log(x / k))
    τ = max(τ, 1.0 / 365.0)
    return min(math.sqrt((10.0 + 2.0 * xk) / τ), 20.0)

@njit(cache=True, inline="always")
def brenner(y, x, k, τ, i, r, /, low, high):
    yτ = intrinsic(x, k, τ, i, r)
    dy = max(y - yτ, 1e-12)
    x = max(x, 1e-12)
    σ = math.sqrt(2.0 * math.pi / max(τ, 1e-12)) * dy / x
    if not math.isfinite(σ): σ = 0.2
    if σ < low: σ = low
    if σ > high: σ = high
    return σ

@njit(cache=True)
def newton(y, x, k, τ, i, r, /, low, high, tol, iters):
    σ = brenner(y, x, k, τ, i, r, low, high)
    for _ in range(iters):
        err = value(x, k, τ, σ, i, r) - y
        if abs(err) <= tol: return σ
        d2ydx2 = vega(x, k, r, σ, τ)
        if not math.isfinite(d2ydx2) or d2ydx2 <= 1e-12: return math.nan
        step = err / d2ydx2
        limit = 0.5 * max(σ, 0.10)
        if step > limit: step = limit
        elif step < -limit: step = -limit
        update = σ - step
        if not math.isfinite(update): return math.nan
        if update <= low: update = 0.5 * (σ + low)
        elif update >= high: update = 0.5 * (σ + high)
        if abs(update - σ) <= 1e-10: return update
        σ = update
    return math.nan

@njit(cache=True)
def bisection(y, x, k, τ, i, r, /, low, high, tol, iters):
    yl = error(y, x, k, τ, low, i, r)  
    yh = error(y, x, k, τ, high, i, r)
    if not math.isfinite(yl) or not math.isfinite(yh): return math.nan
    if yl == 0.0: return low
    if yh == 0.0: return high
    if yl * yh > 0.0: return math.nan
    a, b = low, high
    fa, fb = yl, yh
    for _ in range(iters):
        m = 0.5 * (a + b)
        fm = error(y, x, k, τ, m, i, r)
        if not math.isfinite(fm): return math.nan
        if abs(fm) <= tol or (b - a) <= 1e-10: return m
        if fa * fm <= 0.0: b, fb = m, fm
        else: a, fa = m, fm
    return 0.5 * (a + b)

@njit(cache=True)
def fitted(y, x, k, τ, i, r, /, low, high, tol, iters):
    assert i == 1 or i == -1
    σl, σh = low, high
    for _ in range(iters):
        σml = σl + (σh - σl) / 3.0
        σmh = σh - (σh - σl) / 3.0
        yml = abs(value(x, k, τ, σml, i, r) - y)   
        ymh = abs(value(x, k, τ, σmh, i, r) - y)
        if yml < ymh: σh = σmh
        else: σl = σml
        if (σh - σl) < tol: break
    return 0.5 * (σl + σh)

@njit(cache=True)
def volatility(y, x, k, τ, i, r, /, low, high, tol, iters):
    if not valid(x, k, τ, i) or low <= 0.0 or high <= low: return math.nan
    yl, yh = boundary(x, k, τ, i, r)
    if y < yl - tol or y > yh + tol: return math.nan
    σh = min(high, adaptive(x, k, τ))
    if σh <= low: σh = high
    σ = newton(y, x, k, τ, i, r, low=low, high=σh, tol=tol, iters=iters)
    if math.isfinite(σ): return σ
    return bisection(y, x, k, τ, i, r, low=low, high=σh, tol=tol, iters=max(100, iters))

@njit(cache=True)
def residual(y, x, k, τ, i, r):
    if not valid(x, k, τ, i): return math.nan
    yl, yh = boundary(x, k, τ, i, r)
    if y < yl - 1e-10: return y - yl
    if y > yh + 1e-10: return y - yh
    return 0.0

@njit(cache=True)
def implied_strict(y, x, k, τ, i, r, /, low=1e-4, high=5.0, tol=1e-10, iters=100):
    return volatility(y, x, k, τ, i, r, low=low, high=high, tol=tol, iters=iters)

@njit(cache=True)
def implied_clipped(y, x, k, τ, i, r, /, low=1e-4, high=5.0, tol=1e-10, iters=100):
    if not valid(x, k, τ, i) or low <= 0.0 or high <= low: return math.nan
    yl, yh = boundary(x, k, τ, i, r)
    σh = min(high, adaptive(x, k, τ))
    if y < yl - tol: return low
    if y > yh + tol: return σh
    σ = volatility(y, x, k, τ, i, r, low=low, high=σh, tol=tol, iters=iters)
    if math.isfinite(σ): return σ
    return low if (y - yl) <= (yh - y) else σh

@njit(cache=True)
def implied_fitted(y, x, k, τ, i, r, /, low=1e-4, high=5.0, tol=1e-10, iters=100):
    if not valid(x, k, τ, i) or low <= 0.0 or high <= low: return math.nan
    σh = min(high, adaptive(x, k, τ))
    σ = volatility(y, x, k, τ, i, r, low=low, high=σh, tol=tol, iters=iters)
    if math.isfinite(σ): return σ
    return fitted(y, x, k, τ, i, r, low=low, high=σh, tol=tol, iters=iters)


@njit(cache=True)
def residual_strict(y, x, k, τ, i, r, /, low=1e-4, high=5.0, tol=1e-10, iters=100):
    return residual(y, x, k, τ, i, r)

@njit(cache=True)
def residual_clipped(y, x, k, τ, i, r, /, low=1e-4, high=5.0, tol=1e-10, iters=100):
    return residual(y, x, k, τ, i, r)

@njit(cache=True)
def residual_fitted(y, x, k, τ, i, r, low=1e-4, high=5.0, tol=1e-10, iters=100):
    if not valid(x, k, τ, i) or low <= 0.0 or high <= low: return math.nan
    σ = implied_fitted(y, x, k, τ, i, r, low=low, high=high, tol=tol, iters=iters)
    if not math.isfinite(σ): return math.nan
    return value(x, k, τ, σ, i, r) - y          


class Equations:
    class Greeks: Value, Delta, Gamma, Theta, Rho, Vega, Vomma, Vanna, Charm = value, delta, gamma, theta, rho, vega, vomma, vanna, charm
    class Volatility:
        class Strict: Implied, Residual = implied_strict, residual_strict
        class Clipped: Implied, Residual = implied_clipped, residual_clipped
        class Fitted: Implied, Residual = implied_fitted, residual_fitted



