# -*- coding: utf-8 -*-
"""
Created on Thurs Mar 26 2026
@name:   Technical Objects
@author: Jack Kirby Cook

"""

import pandas as pd

from concepts import Concepts
from support.calculations import Calculation
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = []
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class TechnicalCalculator(Calculation, Logging):
    pctgains = lambda adjusted: adjusted.pct_changes(1)
    netgains = lambda adjusted: adjusted.diff()

    def __call__(self, *args, bars, period, **kwargs):
        assert isinstance(bars, pd.DataFrame)
        technicals = self.calculate(bars, period=period)
        technicals = pd.concat([bars, technicals], axis=1)
        self.alert(technicals)
        return technicals

    def alert(self, technicals):
        instrument = str(Concepts.Securities.Instrument.STOCK).title()
        tickers = "|".join(list(technicals["ticker"].unique()))
        self.console("Calculated", f"{str(instrument)}[{str(tickers)}, {len(technicals):.0f}]")


