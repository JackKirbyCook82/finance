# -*- coding: utf-8 -*-
"""
Created on Tues Mar 24 2026
@name:   Valuation Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from numba import njit

from finance.equations import Equations
from finance.concepts import Concepts
from support.mixins import Logging
from support.concepts import DateRange

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ValuationCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


blackscholes = Equations.Valuation.BlackScholes


@njit(cache=True)
def calculation(x, k, τ, σ, i, r):
    y = np.empty(len(x), dtype=np.float64)  # Black Scholes Valuation
    for idx in range(len(x)):
        y[idx] = blackscholes(x[idx], k[idx], τ[idx], σ[idx], i[idx], r)
    return y


class ValuationCalculator(Logging):
    def __call__(self, options, *args, interest, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if bool(options.empty): return options
        x = options["underlying"].to_numpy(np.float64)  # Market Stock Price
        k = options["strike"].to_numpy(np.float64)  # Option Strike Price
        τ = options["tau"].to_numpy(np.float64)  # Option DTE
        σ = options["volatility"].to_numpy(np.float64)  # Historical Volatility
        i = options["option"].apply(int).to_numpy(np.int8)  # Option Type
        options["valuation"] = calculation(x, k, τ, σ, i, float(interest))
        self.alert(options)
        return options

    def alert(self, dataframe):
        instrument = str(Concepts.Securities.Instrument.OPTION).title()
        tickers = "|".join(list(dataframe["ticker"].unique()))
        expires = DateRange.create(list(dataframe["expire"].unique()))
        expires = f"{expires.minimum.strftime('%Y%m%d')}->{expires.maximum.strftime('%Y%m%d')}"
        self.console("Calculated", f"{str(instrument)}[{str(tickers)}, {str(expires)}, {len(dataframe):.0f}]")



