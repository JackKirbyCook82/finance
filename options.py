# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 2026
@name:   Option Objects
@author: Jack Kirby Cook

"""

import math
import numpy as np
import pandas as pd
from numba import njit, prange
from datetime import date as Date

from finance.concepts import Concepts
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SanityFilter", "ViabilityFilter"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
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


class OptionEquation(object):
    tau = lambda expire: (expire - pd.Timestamp(Date.today())).dt.days
    factor = lambda tau, * interest: 1 / np.exp(tau * interest)
    money = lambda strike, underlying: strike / underlying
    mean = lambda bid, ask, demand, supply: (bid * demand + ask * supply)
    median = lambda bid, ask: (bid + ask) / 2
    spread = lambda bid, ask: ask - bid


class OptionFilter(Logging):
    def __init__(self, *args, criteria, **kwargs):
        assert isinstance(criteria, list)
        super().__init__(*args, **kwargs)
        self.__mask = np.logical_and.reduce(criteria)

    def __call__(self, *args, options, **kwargs):
        assert isinstance(options, pd.DataFrame)
        filtered = options[self.mask]
        self.alert(options, filtered)
        return filtered

    def alert(self, unfiltered, filtered):
        tickers = "|".join(list(unfiltered["ticker"].unique()))
        instrument = str(Concepts.Securities.Instrument.STOCK).title()
        self.console("Filtered", f"{str(instrument)}[{str(tickers)}, {len(unfiltered)|len(filtered)}]")

    @property
    def mask(self): return self.__mask


class SanityFilter(Logging):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        spread = lambda options: options["ask"] > options["bid"]
        supply = lambda options: options["supply"].notna() & (options["supply"] >= 1)
        demand = lambda options: options["demand"].notna() & (options["demand"] >= 1)
        bid = lambda options: options["bid"].notna() & (options["bid"] >= 0)
        ask = lambda options: options["ask"].notna() & (options["ask"] >= 0)
        criteria = [spread, supply, demand, bid, ask]
        super().__init__(*args, criteria=criteria, **kwargs)


class ViabilityFilter(Logging):
    def __init__(self, *args, spread=0.25, size=2, **kwargs):
        assert isinstance(spread, (int, float)) and isinstance(size, int)
        spread = lambda options: (options["ask"] - options["bid"]) * 2 / (options["ask"] + options["bid"]) <= float(spread)
        supply = lambda options: options["supply"] >= int(size)
        demand = lambda options: options["demand"] >= int(size)
        criteria = [spread, supply, demand]
        super().__init__(*args, criteria=criteria, **kwargs)



