# -*- coding: utf-8 -*-
"""
Created on Tues Mar 24 2026
@name:   Implied Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from enum import Enum
from numba import njit
from types import SimpleNamespace
from collections import OrderedDict as ODict

from finance.equations import Equations
from finance.concepts import Concepts
from support.mixins import Logging
from support.concepts import DateRange

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ImpliedCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


Method = Enum("Method", {"STRICT": 0, "CLIPPED": 1, "FITTED": 2})
implied_strict = Equations.Volatility.Strict.Implied
implied_clipped = Equations.Volatility.Clipped.Implied
implied_fitted = Equations.Volatility.Fitted.Implied
residual_strict = Equations.Volatility.Strict.Residual
residual_clipped = Equations.Volatility.Clipped.Residual
residual_fitted = Equations.Volatility.Fitted.Residual


@njit(cache=True)
def calculation(y, x, k, τ, i, r, n, m, /, low=1e-4, high=5.0, tol=1e-10, iters=100):
    σ = np.empty(n, dtype=np.float64)  # Implied Volatility
    ε = np.empty(n, dtype=np.float64)  # Implied Residual

    if m == 0:
        for idx in range(n):
            σ[idx] = implied_strict(y[idx], x[idx], k[idx], τ[idx], i[idx], r, low=low, high=high, tol=tol, iters=iters)
            ε[idx] = residual_strict(y[idx], x[idx], k[idx], τ[idx], i[idx], r, low=low, high=high, tol=tol, iters=iters)
    elif m == 1:
        for idx in range(n):
            σ[idx] = implied_clipped(y[idx], x[idx], k[idx], τ[idx], i[idx], r, low=low, high=high, tol=tol, iters=iters)
            ε[idx] = residual_clipped(y[idx], x[idx], k[idx], τ[idx], i[idx], r, low=low, high=high, tol=tol, iters=iters)
    elif m == 2:
        for idx in range(n):
            σ[idx] = implied_fitted(y[idx], x[idx], k[idx], τ[idx], i[idx], r, low=low, high=high, tol=tol, iters=iters)
            ε[idx] = residual_fitted(y[idx], x[idx], k[idx], τ[idx], i[idx], r, low=low, high=high, tol=tol, iters=iters)
    return σ, ε


class ImpliedCalculator(Logging):
    def __init__(self, *args, method=Method.STRICT, low=1e-4, high=5.0, tol=1e-10, iters=100, **kwargs):
        super().__init__(*args, **kwargs)
        inlet = ODict(y="price", x="underlying", k="strike", τ="tau", i="option", r="interest")
        outlet = ODict(σ="volatility", ε="residual")
        variables = SimpleNamespace(inlet=inlet, outlet=outlet)
        self.__hyperparams = dict(low=low, high=high, tol=tol, iters=iters)
        self.__variables = variables
        self.__method = method

    def __call__(self, *args, options, interest, **kwargs):
        assert isinstance(options, pd.DataFrame)
        y = options["price"].to_numpy(np.float64)
        x = options["underlying"].to_numpy(np.float64)
        k = options["strike"].to_numpy(np.float64)
        τ = options["tau"].to_numpy(np.float64)
        σ = options["volatility"].to_numpy(np.float64)
        i = options["option"].apply(int).to_numpy(np.float64)
        implied = list(calculation(y, x, k, τ, i, float(interest), len(options), int(self.method), **self.hyperparams))
        implied = dict(zip(self.variables.outlet.values(), implied))
        options = pd.concat([options, implied], axis=1)
        self.alert(options)
        return options

    def alert(self, options):
        instrument = str(Concepts.Securities.Instrument.OPTION).title()
        tickers = "|".join(list(options["ticker"].unique()))
        expires = DateRange.create(list(options["expire"].unique()))
        expires = f"{expires.min.strftime('%Y%m%d')}->{expires.max.strftime('%Y%m%d')}"
        self.console("Filtered", f"{str(instrument)}[{str(tickers)}, {str(expires)}, {len(options):.0f}]")

    @property
    def hyperparams(self): return self.__hyperparams
    @property
    def variables(self): return self.__variables
    @property
    def method(self): return self.__method



