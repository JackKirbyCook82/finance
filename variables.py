# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 2024
@name:   Finance Variable Objects
@author: Jack Kirby Cook

"""

import pandas as pd
from abc import ABC
from enum import IntEnum
from datetime import date as Date
from functools import total_ordering
from datetime import datetime as Datetime
from collections import namedtuple as ntuple

from support.dispatchers import typedispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DateRange", "Ticker", "Contract", "Variable", "Actions", "Instruments", "Options", "Positions", "Spreads", "Scenarios", "Technicals", "Securities", "Strategies", "Valuations"]
__copyright__ = "Copyright 2023,SE Jack Kirby Cook"
__license__ = "MIT License"


Actions = IntEnum("Actions", ["OPEN", "CLOSE"], start=1)
Instruments = IntEnum("Instruments", ["PUT", "CALL", "STOCK"], start=1)
Options = IntEnum("Options", ["PUT", "CALL"], start=1)
Positions = IntEnum("Positions", ["LONG", "SHORT"], start=1)
Spreads = IntEnum("Strategy", ["STRANGLE", "COLLAR", "VERTICAL"], start=1)
Valuations = IntEnum("Valuation", ["ARBITRAGE"], start=1)
Scenarios = IntEnum("Scenarios", ["MINIMUM", "MAXIMUM"], start=1)
Technicals = IntEnum("Technicals", ["BARS", "STATISTIC", "STOCHASTIC"], start=1)


class DateRange(ntuple("DateRange", "minimum maximum")):
    def __new__(cls, dates):
        assert isinstance(dates, list)
        assert all([isinstance(date, Date) for date in dates])
        dates = [date if isinstance(date, Date) else date.date() for date in dates]
        return super().__new__(cls, min(dates), max(dates)) if dates else None

    def __contains__(self, date): return self.minimum <= date <= self.maximum
    def __iter__(self): return (date for date in pd.date_range(start=self.minimum, end=self.maximum))
    def __repr__(self): return f"{self.__class__.__name__}({repr(self.minimum)}, {repr(self.maximum)})"
    def __str__(self): return f"{str(self.minimum)}|{str(self.maximum)}"
    def __bool__(self): return self.minimum < self.maximum
    def __len__(self): return (self.maximum - self.minimum).days


class QueryMeta(type):
    def __str__(cls): return str(cls.__name__).lower()
    def __getitem__(cls, string): return cls.fromstring(string)


@total_ordering
class Ticker(ntuple("Ticker", "symbol"), metaclass=QueryMeta):
    def __str__(self): return self.tostring()
    def __eq__(self, other): return str(self.symbol) == str(self.symbol)
    def __lt__(self, other): return str(self.symbol) < str(self.symbol)

    @classmethod
    def fromstring(cls, string): return cls(string)
    def tostring(self): return str(self.symbol)


@total_ordering
class Contract(ntuple("Contract", "symbol expire"), metaclass=QueryMeta):
    def __str__(self): return self.tostring(delimiter="|")
    def __eq__(self, other): return str(self.symbol) == str(other.symbol) and self.expire == other.expire
    def __lt__(self, other): return str(self.symbol) < str(other.symbol) and self.expire < other.expire

    @classmethod
    def fromstring(cls, string, delimiter="_"):
        symbol, expire = str(string).split(delimiter)
        symbol = str(symbol).upper()
        expire = Datetime.strptime(expire, "%Y%m%d")
        return cls(symbol, expire)

    def tostring(self, delimiter="_"):
        symbol = str(self.symbol).upper()
        expire = str(self.expire.strftime("%Y%m%d"))
        contract = list(filter(None, [symbol, expire]))
        return str(delimiter).join(contract)


class Variable(ABC):
    def __str__(self): return "|".join([str(value.name).lower() for value in self if value is not None])
    def __hash__(self): return hash(tuple(self))

    @property
    def title(self): return "|".join([str(string).title() for string in str(self).split("|")])
    @classmethod
    def fields(cls): return list(cls._fields)

    def items(self): return list(zip(self.keys(), self.values()))
    def keys(self): return list(self._fields)
    def values(self): return list(self)


class Security(Variable, ntuple("Security", "instrument position")): pass
class Strategy(Variable, ntuple("Strategy", "spread instrument position")):
    def __new__(cls, spread, instrument, position, *args, **kwargs): return super().__new__(cls, spread, instrument, position)
    def __init__(self, *args, options=[], stocks=[], **kwargs): self.options, self.stocks = options, stocks

    @property
    def securities(self): return list(self.options) + list(self.stocks)


StockLong = Security(Instruments.STOCK, Positions.LONG)
StockShort = Security(Instruments.STOCK, Positions.SHORT)
PutLong = Security(Instruments.PUT, Positions.LONG)
PutShort = Security(Instruments.PUT, Positions.SHORT)
CallLong = Security(Instruments.CALL, Positions.LONG,)
CallShort = Security(Instruments.CALL, Positions.SHORT)
VerticalPut = Strategy(Spreads.VERTICAL, Instruments.PUT, None, options=[PutLong, PutShort])
VerticalCall = Strategy(Spreads.VERTICAL, Instruments.CALL, None, options=[CallLong, CallShort])
CollarLong = Strategy(Spreads.COLLAR, None, Positions.LONG, options=[PutLong, CallShort], stocks=[StockLong])
CollarShort = Strategy(Spreads.COLLAR, None, Positions.SHORT, options=[CallLong, PutShort], stocks=[StockShort])


class VariablesMeta(type):
    def __init_subclass__(mcs, *args, variables, **kwargs): mcs.variables = variables
    def __getitem__(cls, value): return cls.get(value)
    def __iter__(cls): return iter(type(cls).variables)

    @typedispatcher
    def get(cls, key): raise TypeError(type(key).__name__)
    @get.register(tuple)
    def hash(cls, content): return {hash(value): value for value in iter(cls)}[hash(content)]
    @get.register(str)
    def string(cls, string): return {str(value): value for value in iter(cls)}[str(string).lower()]


class SecuritiesMeta(VariablesMeta, variables=[StockLong, StockShort, PutLong, PutShort, CallLong, CallShort]):
    def security(cls, instrument, position): return Securities[(cls.instrument(instrument), cls.position(position))]

    @staticmethod
    def instrument(instrument): return Instruments[str(instrument).upper()]
    @staticmethod
    def position(position): return Positions[str(position).upper()]

    @property
    def Stocks(cls): return iter([StockLong, StockShort])
    @property
    def Options(cls): return iter([PutLong, PutShort, CallLong, CallShort])
    @property
    def Puts(cls): return iter([PutLong, PutShort])
    @property
    def Calls(cls): return iter([CallLong, CallShort])

    class Stock:
        Long = StockLong
        Short = StockShort
    class Option:
        class Put:
            Long = PutLong
            Short = PutShort
        class Call:
            Long = CallLong
            Short = CallShort
    class Long:
        Stock = StockLong
        Put = PutLong
        Call = CallLong
    class Short:
        Stock = StockShort
        Put = PutShort
        Call = CallShort


class StrategiesMeta(VariablesMeta, variables=[CollarLong, CollarShort, VerticalPut, VerticalCall]):
    def strategy(cls, spread, instrument, position): return Strategies[(cls.spread(spread), cls.instrument(instrument), cls.position(position))]

    @staticmethod
    def spread(spread): return Spreads[str(spread).upper()]
    @staticmethod
    def instrument(instrument): return Instruments[str(instrument).upper()]
    @staticmethod
    def position(position): return Positions[str(position).upper()]

    @property
    def Collars(cls): return iter([CollarLong, CollarShort])
    @property
    def Verticals(cls): return iter([VerticalPut, VerticalCall])

    class Collar:
        Long = CollarLong
        Short = CollarShort
    class Vertical:
        Put = VerticalPut
        Call = VerticalCall


class Securities(object, metaclass=SecuritiesMeta): pass
class Strategies(object, metaclass=StrategiesMeta): pass



