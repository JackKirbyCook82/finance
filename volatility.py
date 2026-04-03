# -*- coding: utf-8 -*-
"""
Created on Tues Mar 24 2026
@name:   Volatility Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from numba import njit
from types import SimpleNamespace
from collections import OrderedDict as ODict

from finance.equations import Equations
from finance.concepts import Concepts
from support.mixins import Logging
from support.concepts import DateRange

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["VolatilityCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


implied = Equations.Volatility.Implied


@njit(cache=True)
def calculation(y, x, k, τ, i, r, n, /, low=1e-4, high=5.0, tol=1e-10, iters=100):
    σ = np.empty(n, dtype=np.float64)  # Implied Volatility
    for idx in range(n):
        σ[idx] = implied(y[idx], x[idx], k[idx], τ[idx], i[idx], r, low=low, high=high, tol=tol, iters=iters)
    return σ


class VolatilityCalculator(Logging):
    def __init__(self, *args, low=1e-4, high=5.0, tol=1e-10, iters=100, **kwargs):
        super().__init__(*args, **kwargs)
        inlet = ODict(y="price", x="underlying", k="strike", τ="tau", i="option", r="interest")
        outlet = ODict(σ="implied")
        variables = SimpleNamespace(inlet=inlet, outlet=outlet)
        self.__hyperparams = dict(low=low, high=high, tol=tol, iters=iters)
        self.__variables = variables

    def __call__(self, options, *args, interest, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if bool(options.empty): return options
        y = options["median"].to_numpy(np.float64)  # Market Option Price
        x = options["underlying"].to_numpy(np.float64)  # Market Stock Price
        k = options["strike"].to_numpy(np.float64)  # Option Strike
        τ = options["tau"].to_numpy(np.float64)  # Option DTE
        i = options["option"].apply(int).to_numpy(np.float64)  # Option Type
        volatility = [calculation(y, x, k, τ, i, float(interest), len(options), **self.hyperparams)]
        volatility = dict(zip(self.variables.outlet.values(), volatility))
        options = pd.concat([options, pd.DataFrame(volatility)], axis=1)
        self.alert(options)
        return options

    def alert(self, dataframe):
        instrument = str(Concepts.Securities.Instrument.OPTION).title()
        tickers = "|".join(list(dataframe["ticker"].unique()))
        expires = DateRange.create(list(dataframe["expire"].unique()))
        expires = f"{expires.minimum.strftime('%Y%m%d')}->{expires.maximum.strftime('%Y%m%d')}"
        self.console("Calculated", f"{str(instrument)}[{str(tickers)}, {str(expires)}, {len(dataframe):.0f}]")

    @property
    def hyperparams(self): return self.__hyperparams
    @property
    def variables(self): return self.__variables



