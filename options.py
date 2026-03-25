# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 2026
@name:   Option Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from datetime import date as Date

from concepts import Concepts
from support.calculations import Calculation
from support.concepts import DateRange
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SanityFilter", "ViabilityFilter", "OptionCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


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
        instrument = str(Concepts.Securities.Instrument.OPTION).title()
        tickers = "|".join(list(unfiltered["ticker"].unique()))
        expires = DateRange.create(list(unfiltered["expire"].unique()))
        expires = f"{expires.min.strftime('%Y%m%d')}->{expires.max.strftime('%Y%m%d')}"
        self.console("Filtered", f"{str(instrument)}[{str(tickers)}, {str(expires)}, {len(unfiltered):.0f}|{len(filtered):.0f}]")

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


class OptionCalculator(Calculation, Logging):
    tau = lambda expire: (expire - pd.Timestamp(Date.today())).dt.days / 365
    discount = lambda tau, *, interest: 1 / np.exp(tau * interest)
    intrinsic = lambda strike, underlying, option: max((underlying - strike) * int(option), 0) * int(option)
    moneyness = lambda strike, underlying: strike / underlying
    mean = lambda bid, ask, demand, supply: (bid * demand + ask * supply)
    median = lambda bid, ask: (bid + ask) / 2
    spread = lambda bid, ask: ask - bid

    def __call__(self, *args, options, interest, **kwargs):
        assert isinstance(options, pd.DataFrame)
        options = self.calculate(options, interest=interest)
        self.alert(options)
        return options

    def alert(self, options):
        instrument = str(Concepts.Securities.Instrument.OPTION).title()
        tickers = "|".join(list(options["ticker"].unique()))
        expires = DateRange.create(list(options["expire"].unique()))
        expires = f"{expires.min.strftime('%Y%m%d')}->{expires.max.strftime('%Y%m%d')}"
        self.console("Filtered", f"{str(instrument)}[{str(tickers)}, {str(expires)}, {len(options):.0f}]")



