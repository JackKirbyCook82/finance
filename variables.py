# -*- coding: utf-8 -*-
"""
Created on Weds May 27 2026
@name:   Finance Variable Objects
@author: Jack Kirby Cook
@file:   finance/osi.py

"""

import regex as re
import pandas as pd
from decimal import Decimal
from datetime import date as Date
from datetime import datetime as Datetime
from dataclasses import dataclass, asdict, fields

from finance.enumerations import Right

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["OSI", "Settlement", "Settlements"]
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


# @dataclass(frozen=True, slots=True)
# class Settlement: expire: Date; strike: float
# class Settlements(list):
#     def create(self, expires, /, zscore, today, spot, volatility, interest, dividends):
#         assert isinstance(expires, list) and all([isinstance(expire, Date) for expire in expires])
#         tau = lambda expire: (expire - today).days / 252
#         forward = lambda expire: spot * np.exp(np.subtract(interest, dividends) * tau(expire))
#         strike = lambda z, expire: forward(expire) * np.exp(zscore * index * volatility * np.sqrt(tau(expire)))
#         expires = [SimpleNamespace(expire=expire, tau=tau(expire), forward=forward(forward)) for expire in expires]
#         strikes = [SimpleNamespace(expire=expires, ) for expire in expires]
#         strikes = NumberRange([forward * np.exp(zscore * index * volatility * np.sqrt(tau)) for index in (-1, +1)])


