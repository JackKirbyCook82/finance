# -*- coding: utf-8 -*-
"""
Created on Weds May 27 2026
@name:   Finance OSI Objects
@author: Jack Kirby Cook

"""

import regex as re
import pandas as pd
from enum import Enum
from decimal import Decimal
from datetime import date as Date
from datetime import datetime as Datetime
from dataclasses import dataclass, asdict, fields

from finance.enumerations import Option

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["OSI"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class OSIError(Exception): pass
class OSIParseError(OSIError): pass
class OSICreateError(OSIError): pass
class OSIEmptyError(OSIError): pass

class OSIMeta(type):
    def __call__(cls, contents):
        try: contents = dict(contents.items())
        except AttributeError: pass
        if isinstance(contents, pd.Series): contents = contents.to_dict()
        if isinstance(contents, dict): instance = super().__call__(**{field.name: contents[field.name] for field in fields(cls)})
        elif isinstance(contents, str): instance = super().__call__(*cls.parse(contents))
        elif isinstance(contents, (list, tuple)): instance = super().__call__(*contents)
        else: raise OSICreateError()
        if instance.option == Option.EMPTY: raise OSIEmptyError()
        return instance

    @staticmethod
    def parse(contents):
        pattern = r"^(?P<ticker>[A-Z]+)(?P<expire>\d{6})(?P<option>[PC])(?P<strike>\d{8})$"
        match = re.search(pattern, contents)
        if not match: raise OSIParseError()
        values = match.groupdict()
        ticker = values["ticker"].upper()
        expire = Datetime.strptime(values["expire"], "%y%m%d").date()
        option = {"P": Option.PUT, "C": Option.CALL}[values["option"]]
        strike = float(Decimal(values["strike"]) / Decimal("1000"))
        return [ticker, expire, option, strike]


@dataclass(frozen=True)
class OSI(metaclass=OSIMeta):
    ticker: str; expire: Date; option: Enum; strike: float

    def __str__(self):
        ticker = self.ticker.upper()
        expire = self.expire.strftime("%y%m%d")
        if self.option == Option.PUT: option = "P"
        elif self.option == Option.CALL: option = "C"
        else: raise ValueError(self.option)
        strike = int((Decimal(self.strike) * Decimal("1000")).to_integral_value())
        strike = f"{strike:08d}"
        return f"{ticker}{expire}{option}{strike}"

    def items(self): return asdict(self).items()
    def values(self): return asdict(self).values()
    def keys(self): return asdict(self).keys()
