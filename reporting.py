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
from types import SimpleNamespace
from abc import ABC, abstractmethod

from finance.enumerations import Instrument
from support.meta import AttributeMeta
from support.decorators import Dispatchers
from support.custom import DateRange, NumberRange

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Results", "Analysis"]
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


class Results(ABC):
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
    def contract(self, contracts, *args, **kwargs):
        if isinstance(contracts, list):
            tickers = list(set([contract.ticker for contract in contracts]))
            expires = DateRange(list(set([contract.expire for contract in contracts])))
        else: raise TypeError(type(contracts))
        return Scope(instrument=Instrument.CONTRACT, tickers=tickers, expires=expires)

    @scope.register(Instrument.SPREAD)
    def spread(self, prospects, *args, **kwargs):
        if isinstance(prospects, list):
            tickers = list(set([prospect.ticker for prospect in prospects]))
            expires = DateRange(list(set([expire for prospect in prospects for expire in iter(prospect.expires)])))
        else: raise TypeError(type(prospects))
        return Scope(instrument=Instrument.SPREAD, tickers=tickers, expires=expires)

    @staticmethod
    def results(scope, size):
        assert isinstance(scope, Scope)
        if isinstance(size, int): size = f"{size:.0f}"
        elif isinstance(size, tuple):
            assert len(size) == 2
            before, after = size
            size = f"{int(before):.0f}|{int(after):.0f}, {after / before * 100:.0f}%"
        instrument = str(scope.instrument).title()
        return f"{str(instrument)}[{str(scope)}, {str(size)}]"


class Analysis(ABC, metaclass=AttributeMeta):
    @abstractmethod
    def analysis(self, contents): pass
    @abstractmethod
    def survival(self, contents): pass
    @property
    @abstractmethod
    def metrics(self): pass
    @staticmethod
    @abstractmethod
    def boundary(contents): pass


class Targets(Analysis, ABC, attribute="Targets"):
    def analysis(self, targets):
        boundary = self.boundary(targets)
        survival = self.survival(targets)
        zspread = f"|ZSpread| >= {self.metrics.zspread:.2f} [{boundary.zspreads.minimum:+.2f} -> {boundary.zspreads.maximum:+.2f}, {survival.zspreads:.0f}%]"
        multiple = f"Multiple >= {self.metrics.multiple:.2f} [{boundary.multiples.minimum:+.2f} -> {boundary.multiples.maximum:+.2f}, {survival.multiples:.0f}%]"
        ratio = f"Ratio >= {self.metrics.ratio:.2f} [{boundary.ratios.minimum:+.2f} -> {boundary.ratios.maximum:+.2f}, {survival.ratios:.0f}%]"
        return [zspread, multiple, ratio]

    def survival(self, targets):
        zspreads = [target.zspread >= self.metrics.zspread for target in targets]
        multiples = [target.multiple >= self.metrics.multiple for target in targets]
        ratios = [target.ratio >= self.metrics.ratio for target in targets]
        zspreads = sum(zspreads) / len(zspreads) * 100
        multiples = sum(multiples) / len(multiples) * 100
        ratios = sum(ratios) / len(ratios) * 100
        return SimpleNamespace(zspreads=zspreads, multiples=multiples, ratios=ratios)

    @staticmethod
    def boundary(targets):
        zspreads = [target.zspread for target in targets]
        multiples = [target.multiple for target in targets]
        ratios = [target.ratio for target in targets]
        zspreads = NumberRange([min(zspreads), max(zspreads)])
        multiples = NumberRange([min(multiples), max(multiples)])
        ratios = NumberRange([min(ratios), max(ratios)])
        return SimpleNamespace(zspreads=zspreads, multiples=multiples, ratios=ratios)


class Viability(Analysis, ABC, attribute="Viability"):
    def analysis(self, options):
        boundary = self.boundary(options)
        survival = self.survival(options)
        moneyness = f"|Moneyness| <= {self.metrics.moneyness:.2f} [{boundary.moneyness.minimum:+.2f} -> {boundary.moneyness.maximum:+.2f}, {survival.moneyness:.0f}%]"
        tightness = f"Tightness <= {self.metrics.tightness:.2f} [{boundary.tightness.minimum:+.2f} -> {boundary.tightness.maximum:+.2f}, {survival.tightness:.0f}%]"
        activity = f"Activity >= {self.metrics.activity:.2f} [{boundary.activity.minimum:+.2f} -> {boundary.activity.maximum:+.2f}, {survival.activity:.0f}%]"
        return [moneyness, tightness, activity]

    def survival(self, options):
        moneyness = (options["moneyness"].abs() <= self.metrics.moneyness).sum() / len(options.index) * 100
        tightness = (options["tightness"] <= self.metrics.tightness).sum() / len(options.index) * 100
        activity = (options["activity"] >= self.metrics.activity).sum() / len(options.index) * 100
        return SimpleNamespace(moneyness=moneyness, tightness=tightness, activity=activity)

    @staticmethod
    def boundary(options):
        options = options[["moneyness", "tightness", "activity"]]
        moneyness = NumberRange(options["moneyness"].to_list())
        tightness = NumberRange(options["tightness"].to_list())
        activity = NumberRange(options["activity"].to_list())
        return SimpleNamespace(moneyness=moneyness, tightness=tightness, activity=activity)





