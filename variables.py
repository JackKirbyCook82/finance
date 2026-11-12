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

from finance.enumerations import Enumerations
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["OSI", "Categories", "Quantities"]
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
        right = {str(right).upper()[0]: right for right in Enumerations.Right}[values["right"]]
        strike = float(Decimal(values["strike"]) / Decimal("1000"))
        return [ticker, expire, right, strike]


@dataclass(frozen=True)
class OSI(metaclass=OSIMeta):
    ticker: str; expire: Date; right: Enumerations.Right; strike: float

    def __str__(self):
        ticker = self.ticker.upper()
        expire = self.expire.strftime("%y%m%d")
        right = {right: str(right).upper()[0] for right in Enumerations.Right}[self.right]
        strike = int((Decimal(self.strike) * Decimal("1000")).to_integral_value())
        strike = f"{strike:08d}"
        return f"{ticker}{expire}{right}{strike}"

    def items(self): return asdict(self).items()
    def values(self): return asdict(self).values()
    def keys(self): return asdict(self).keys()


class CategoryError(Exception, metaclass=RegistryMeta): pass
class CategoryTickerError(CategoryError, register="ticker"): pass
class CategoryExpireError(CategoryError, register="expire"): pass
class CategoryStrikeError(CategoryError, register="strike"): pass
class ConceptSettlementError(CategoryError, register="settlement"): pass


class Category(frozenset):
    def __init_subclass__(cls, /, category, **kwargs): cls.category = category
    def __new__(cls, contents):
        if isinstance(contents, pd.DataFrame): contents = set(contents[cls.category].unique())
        elif isinstance(contents, pd.Series): contents = set(contents.unique())
        elif isinstance(contents, list): contents = set([getattr(content, cls.category) for content in contents])
        else: raise CategoryError[cls.category]()
        return super().__new__(cls, contents)


class Ticker(Category):
    def __str__(self): return f"{','.join(list(self))}"

class Settlement(Category):
    def __str__(self): return f"{','.join(list(self))}"

class Expire(Category):
    def __str__(self):
        if min(self) == max(self): return f"{min(self).strftime('%Y/%m/%d')}"
        return f"{min(self).strftime('%Y/%m/%d')}->{max(self).strftime('%Y/%m/%d')}"

class Strike(Category):
    def __str__(self):
        if min(self) == max(self): return f"{min(self):.2}"
        return f"{min(self):.2f}->{max(self):.2f}"


@dataclass
class Quantity:
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


class Moneyness(Quantity, upward=False, absolute=True): pass
class Tightness(Quantity, upward=False): pass
class Activity(Quantity, upward=True): pass
class ZSpread(Quantity, upward=True): pass
class Multiple(Quantity, upward=True): pass
class Ratio(Quantity, upward=True): pass


class Quantities:
    Moneyness = Moneyness
    Tightness = Tightness
    Activity = Activity
    ZSpread = ZSpread
    Multiple = Multiple
    Ratio = Ratio

class Categories:
    Ticker = Ticker
    Expire = Expire
    Strike = Strike
    Settlement = Settlement

