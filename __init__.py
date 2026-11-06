# -*- coding: utf-8 -*-
"""
Created on Weds May 27 2026
@name:   Finance Variable Objects
@author: Jack Kirby Cook
@file:   finance/logging.py

"""

import pandas as pd
from abc import ABC, abstractmethod

from finance.enumerations import Instrument
from support.decorators import Dispatchers
from support.custom import DateRange

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Scope", "Results", "Analysis", "Empty"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class Empty(Exception):
    pass


class Scope(ABC):
    def scope(self, contents):
        strings = list()
        try:
            tickers = self.tickers(contents)
            if bool(tickers):
                tickers = list(tickers)
                tickers = "|".join(list(tickers))
                strings.append("|".join(list(tickers)))
        except AttributeError: pass
        try:
            expires = self.expires(contents)
            if bool(expires):
                expires = DateRange(expires)
                expires = f"{expires.minimum.strftime('%Y%m%d')}->{expires.maximum.strftime('%Y%m%d')}"
                strings.append(expires)
        except AttributeError: pass
        return ", ".join(strings)

    @Dispatchers.Value(locator=0)
    def tickers(self, contents): raise TypeError(type(contents))

    @tickers.register(pd.DataFrame)
    def _(self, dataframe):
        if "ticker" not in dataframe.columns: return set()
        return set(dataframe["ticker"].dropna().unique())

    @tickers.register(list)
    def _(self, contents):
        return set([content.ticker for content in contents])

    @Dispatchers.Value(locator=0)
    def expires(self, contents): raise TypeError(type(contents))

    @expires.register(pd.DataFrame)
    def _(self, dataframe):
        if "expire" not in dataframe.columns: return set()
        return set(dataframe["expire"].dropna().unique())

    @expires.register(list)
    def _(self, contents):
        try: return set([content.expire for content in contents])
        except AttributeError: return set([expire for content in contents for expire in iter(content.expires)])


class Results(ABC):
    @staticmethod
    def results(scope, size, classification):
        if isinstance(size, int): size = f"{size:.0f}"
        elif isinstance(size, tuple):
            assert len(size) == 2
            before, after = size
            size = f"{int(before):.0f}|{int(after):.0f}, {after / before * 100:.0f}%"
        return f"{str(classification).title()}[{str(scope)}, {str(size)}]"


class Analysis(ABC):
    def analysis(self, contents):
        boundary = self.boundary(contents)
        survival = self.survival(contents)
        reports = self.reports(boundary, survival)
        assert isinstance(reports, list)
        assert all([isinstance(report, str) for report in reports])
        return reports

    @staticmethod
    @abstractmethod
    def boundary(contents): pass
    @abstractmethod
    def survival(self, contents): pass
    @abstractmethod
    def reports(self, boundary, survival): pass



