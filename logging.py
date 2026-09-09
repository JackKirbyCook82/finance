# -*- coding: utf-8 -*-
"""
Created on Weds May 27 2026
@name:   Finance Variable Objects
@author: Jack Kirby Cook
@file:   finance/logging.py

"""

import pandas as pd
from typing import Optional
from dataclasses import dataclass

from finance.enumerations import Instrument
from support.decorators import Dispatchers
from support.custom import DateRange
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Logging"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True, slots=True)
class Scope:
    instrument: Instrument; tickers: list; expires: Optional[DateRange] = None

    def __str__(self):
        if self.expires is not None:
            tickers = "|".join(self.tickers)
            expires = f"{self.expires.minimum.strftime('%Y%m%d')}->{self.expires.maximum.strftime('%Y%m%d')}"
            return ", ".join([tickers, expires])
        else: return "|".join(self.tickers)


class Logging(Logging):
    @Dispatchers.Value(locator="instrument")
    def scope(self, contents, *args, instrument, **kwargs): raise ValueError(instrument)

    @scope.register(Instrument.STOCK)
    def stock(self, contents, *args, **kwargs):
        if isinstance(contents, pd.DataFrame):
            tickers = list(contents["ticker"].unique())
        elif isinstance(contents, list):
            tickers = list(set([symbol.ticker for symbol in contents]))
        else: raise TypeError(type(contents))
        return Scope(instrument=Instrument.STOCK, tickers=tickers)

    @scope.register(Instrument.OPTION)
    def option(self, contents, *args, **kwargs):
        if isinstance(contents, pd.DataFrame):
            tickers = list(contents["ticker"].unique())
            expires = DateRange(list(contents["expire"].unique()))
        elif isinstance(contents, list):
            tickers = list(set([symbol.ticker for symbol in contents]))
            expires = DateRange(list(set([contract.expire for contract in contents])))
        else: raise TypeError(type(contents))
        return Scope(instrument=Instrument.OPTION, tickers=tickers, expires=expires)

    @scope.register(Instrument.CONTRACT)
    def contract(self, contents, *args, **kwargs):
        if isinstance(contents, list):
            tickers = list(set([symbol.ticker for symbol in contents]))
            expires = DateRange(list(set([contract.expire for contract in contents])))
        else: raise TypeError(type(contents))
        return Scope(instrument=Instrument.CONTRACT, tickers=tickers, expires=expires)

    @scope.register(Instrument.SPREAD)
    def spread(self, contents, *args, **kwargs):
        if isinstance(contents, list):
            tickers = list(set([symbol.ticker for symbol in contents]))
            expires = DateRange(list(set([contract.expire for contract in contents])))
        else: raise TypeError(type(contents))
        return Scope(instrument=Instrument.SPREAD, tickers=tickers, expires=expires)

    def results(self, *args, scope, size, title, **kwargs):
        assert isinstance(scope, Scope)
        if isinstance(size, int): size = f"{size:.0f}"
        elif isinstance(size, tuple):
            assert len(size) == 2
            before, after = size
            size = f"{int(before):.0f}|{int(after):.0f}, {after / before * 100:.0f}%"
        instrument = str(scope.instrument).title()
        strings = kwargs.get("strings", [])
        self.console(str(title), f"{str(instrument)}[{str(scope)}, {str(size)}]")
        for string in strings:
            self.console(str(title), str(string))





