# -*- coding: utf-8 -*-
"""
Created on Weds May 27 2026
@name:   Finance Variables Objects
@author: Jack Kirby Cook
@file:   finance/options.py

"""

import regex as re
import numpy as np
import pandas as pd
from decimal import Decimal
from numpy.typing import NDArray
from datetime import date as Date
from functools import cached_property
from datetime import datetime as Datetime
from dataclasses import dataclass, asdict, fields

from finance.enumerations import Right
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["OSI", "Tickers", "Expires", "Strikes", "Moneyness", "Tightness", "Activity", "ZSpread", "Multiple"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class OSIError(Exception): pass
class OSIParseError(OSIError): pass
class OSICreateError(OSIError): pass
class OSIEmptyError(OSIError): pass


class OSIMeta(type):
    def __call__(cls, contents):
        if isinstance(contents, cls): return contents
        if isinstance(contents, pd.Series): contents = contents.to_dict()
        elif hasattr(contents, "items"): contents = dict(contents.items())
        if isinstance(contents, dict): instance = super().__call__(**{field.name: contents[field.name] for field in fields(cls)})
        elif isinstance(contents, str): instance = super().__call__(*cls.parse(contents))
        elif isinstance(contents, (list, tuple)): instance = super().__call__(*contents)
        else: raise OSICreateError()
        return instance

    @staticmethod
    def parse(contents):
        pattern = r"^(?P<ticker>[A-Z]+)(?P<expire>\d{6})(?P<right>[PC])(?P<strike>\d{8})$"
        match = re.search(pattern, contents)
        if not match: raise OSIParseError()
        values = match.groupdict()
        ticker = values["ticker"].upper()
        expire = Datetime.strptime(values["expire"], "%y%m%d").date()
        right = {str(right).upper()[0]: right for right in Right}[values["right"]]
        strike = float(Decimal(values["strike"]) / Decimal("1000"))
        return [ticker, expire, right, strike]


@dataclass(frozen=True)
class OSI(metaclass=OSIMeta):
    ticker: str; expire: Date; right: Right; strike: float

    def __str__(self):
        ticker = self.ticker.upper()
        expire = self.expire.strftime("%y%m%d")
        right = {right: str(right).upper()[0] for right in Right}[self.right]
        strike = int((Decimal(self.strike) * Decimal("1000")).to_integral_value())
        strike = f"{strike:08d}"
        return f"{ticker}{expire}{right}{strike}"

    def items(self): return asdict(self).items()
    def values(self): return asdict(self).values()
    def keys(self): return asdict(self).keys()


class ConceptError(Exception, metaclass=RegistryMeta): pass
class ConceptTickerError(ConceptError, register="tickers"): pass
class ConceptExpireError(ConceptError, register="expires"): pass
class ConceptStrikeError(ConceptError, register="strikes"): pass


class Categorial(frozenset):
    def __new__(cls, contents):
        concept = str(cls.__name__.lower())
        if isinstance(contents, pd.DataFrame): contents = set(contents[concept].unique())
        elif isinstance(contents, pd.Series): contents = set(contents.unique())
        elif isinstance(contents, list): contents = set([content.ticker for content in contents])
        elif isinstance(contents, (Date, float, str)): contents = {contents}
        else: raise ConceptError[concept]()
        return super().__new__(cls, contents)


class Tickers(Categorial):
    def __str__(self): return f"{','.join(list(self))}"

class Expires(Categorial):
    def __str__(self):
        if min(self) == max(self): return f"{min(self).strftime('%Y/%m/%d')}"
        return f"{min(self).strftime('%Y/%m/%d')}->{max(self).strftime('%Y/%m/%d')}"

class Strikes(Categorial):
    def __str__(self):
        if min(self) == max(self): return f"{min(self):.2}"
        return f"{min(self):.2f}->{max(self):.2f}"


@dataclass
class Quantitative:
    values: NDArray[np.float32] | list[object]; metric: float

    def __init_subclass__(cls, /, upward, absolute=False, **kwargs):
        assert isinstance(upward, bool) and isinstance(absolute, bool)
        cls.upward = upward
        cls.absolute = absolute

    def __post_init__(self):
        variable = str(self.__class__.__name__.lower())
        if isinstance(self.values, list): self.values = np.array([getattr(value, variable) for value in self.values])
        if not self.absolute: pass
        else: self.values = np.abs(self.values)

    def __str__(self):
        name = str(self.__class__.__name__)
        title = f"|{str(name)}|" if self.absolute else str(name)
        sign = ">=" if self.upward else "<="
        return f"{title} {sign} {self.metric:.2f} [{self.minimum:+.2f} -> {self.maximum:+.2f}, {self.percentage:.0f}%]"

    @cached_property
    def percentage(self):
        if bool(self.upward): return (self.values >= self.metric).sum() / len(self.values) * 100
        else: return (self.values <= self.metric).sum() / len(self.values) * 100

    @property
    def maximum(self): return self.values.max()
    @property
    def minimum(self): return self.values.min()


class Moneyness(Quantitative, upward=False, absolute=True): pass
class Tightness(Quantitative, upward=False): pass
class Activity(Quantitative, upward=True): pass
class ZSpread(Quantitative, upward=True): pass
class Multiple(Quantitative, upward=True): pass
class Ratio(Quantitative, upward=True): pass



