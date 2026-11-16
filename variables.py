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
from abc import ABC
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
__all__ = ["Variables", "Collections", "Metrics", "OSI"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class VariableError(Exception, metaclass=RegistryMeta): pass
class TickerError(VariableError, register="ticker"): pass
class ExpireError(VariableError, register="expire"): pass
class StrikeError(VariableError, register="strike"): pass
class MoneynessError(VariableError, register="moneyness"): pass
class TightnessError(VariableError, register="tightness"): pass
class ActivityError(VariableError, register="activity"): pass
class ZScoreError(VariableError, register="zscore"): pass
class MultipleError(VariableError, register="multiple"): pass
class RatioError(VariableError, register="ratio"): pass


class Variable(ABC): pass
class StrVariable(str, Variable):
    def __str__(self): return self.upper()

class NumVariable(float, Variable):
    def __str__(self): return f"{self:.2f}"
    def __float__(self): return float(self)

class DateVariable(Date, Variable):
    def __str__(self): return f"{self.strftime('%Y/%m/%d')}"


class TickerVariable(StrVariable): pass
class ExpireVariable(DateVariable): pass
class StrikeVariable(NumVariable): pass
class MoneynessVariable(NumVariable): pass
class TightnessVariable(NumVariable): pass
class ActivityVariable(NumVariable): pass
class ZScoreVariable(NumVariable): pass
class MultipleVariable(NumVariable): pass
class RatioVariable(NumVariable): pass


class Variables:
    Ticker = TickerVariable
    Expire = ExpireVariable
    Strike = StrikeVariable
    Moneyness = MoneynessVariable
    Tightness = TightnessVariable
    Activity = ActivityVariable
    ZScore = ZScoreVariable
    Multiple = MultipleVariable
    Ratio = RatioVariable


class Collections:
    pass


class Metrics:
    pass


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




