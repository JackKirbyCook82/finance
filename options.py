# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 2026
@name:   Option Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from abc import ABC
from datetime import date as Date

from finance.concepts import Concepts
from support.calculations import Calculation
from support.concepts import DateRange
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SanityFilter", "ViabilityFilter", "OptionCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class OptionFilter(Calculation, Logging, ABC):
    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if not bool(options): return options
        mask = self.calculate(options, *args, **kwargs)
        filtered = options[mask]
        self.alert(options, len(options), len(filtered))
        return filtered

    def alert(self, dataframe, previous, post):
        instrument = str(Concepts.Securities.Instrument.OPTION).title()
        tickers = "|".join(list(dataframe["ticker"].unique()))
        expires = DateRange.create(list(dataframe["expire"].unique()))
        expires = f"{expires.minimum.strftime('%Y%m%d')}->{expires.maximum.strftime('%Y%m%d')}"
        self.console("Filtered", f"{str(instrument)}[{str(tickers)}, {str(expires)}, {int(previous):.0f}|{int(post):.0f}]")


class SanityFilter(OptionFilter, variables=["sanity"]):
    sanity = lambda spread, supply, demand, bid, ask: np.logical_and.reduce([spread, supply, demand, bid, ask])
    spread = lambda options: options["ask"] > options["bid"]
    supply = lambda options: options["supply"].notna() & (options["supply"] >= 1)
    demand = lambda options: options["demand"].notna() & (options["demand"] >= 1)
    bid = lambda options: options["bid"].notna() & (options["bid"] >= 0)
    ask = lambda options: options["ask"].notna() & (options["ask"] >= 0)


class ViabilityFilter(OptionFilter, variables=["viability"]):
    viability = lambda spread, supply, demand:  np.logical_and.reduce([spread, supply, demand])
    spread = lambda options, *, spread=0.25: (options["ask"] - options["bid"]) * 2 / (options["ask"] + options["bid"]) <= float(spread)
    supply = lambda options, *, size=2: options["supply"] >= int(size)
    demand = lambda options, *, size=2: options["demand"] >= int(size)


class OptionCalculator(Calculation, Logging):
    tau = lambda expire: (pd.to_datetime(expire) - pd.Timestamp(Date.today())).dt.days / 365
    discount = lambda tau, *, interest: 1 / np.exp(tau * interest)
    intrinsic = lambda strike, underlying, option: (np.maximum((underlying - strike) * option.astype(int), 0) * option.astype(int))
    moneyness = lambda strike, underlying: strike / underlying
    pricing = lambda mean, median: (mean + median) / 2
    mean = lambda bid, ask, demand, supply: (bid * demand + ask * supply)
    median = lambda bid, ask: (bid + ask) / 2
    spread = lambda bid, ask: ask - bid

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if bool(options.empty): return options
        calculated = self.calculate(options, *args, **kwargs)
        calculated = pd.concat([options, calculated], axis=1)
        return calculated

    def alert(self, dataframe):
        instrument = str(Concepts.Securities.Instrument.OPTION).title()
        tickers = "|".join(list(dataframe["ticker"].unique()))
        expires = DateRange.create(list(dataframe["expire"].unique()))
        expires = f"{expires.min.strftime('%Y%m%d')}->{expires.max.strftime('%Y%m%d')}"
        self.console("Calculated", f"{str(instrument)}[{str(tickers)}, {str(expires)}, {len(dataframe):.0f}]")





