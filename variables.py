# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 2024
@name:   Finance Variable Objects
@author: Jack Kirby Cook

"""

from enum import IntEnum
from datetime import date as Date
from functools import total_ordering
from collections import namedtuple as ntuple

from support.dispatchers import typedispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DateRange", "Query", "Contract", "Instruments", "Positions", "Options", "Securities", "Spreads", "Strategies", "Valuations", "Scenarios"]
__copyright__ = "Copyright 2023,SE Jack Kirby Cook"
__license__ = "MIT License"


Instruments = IntEnum("Instruments", ["PUT", "CALL", "STOCK"], start=1)
Options = IntEnum("Options", ["PUT", "CALL"], start=1)
Positions = IntEnum("Positions", ["LONG", "SHORT"], start=1)
Spreads = IntEnum("Strategy", ["STRANGLE", "COLLAR", "VERTICAL"], start=1)
Valuations = IntEnum("Valuation", ["ARBITRAGE"], start=1)
Scenarios = IntEnum("Scenarios", ["MARTINGALE", "MINIMUM", "MAXIMUM"], start=1)


class DateRange(ntuple("DateRange", "minimum maximum")):
    def __new__(cls, dates):
        assert isinstance(dates, list)
        assert all([isinstance(date, Date) for date in dates])
        dates = [date if isinstance(date, Date) else date.date() for date in dates]
        return super().__new__(cls, min(dates), max(dates)) if dates else None

    def __contains__(self, date): return self.minimum <= date <= self.maximum
    def __repr__(self): return f"{self.__class__.__name__}({repr(self.minimum)}, {repr(self.maximum)})"
    def __str__(self): return f"{str(self.minimum)}|{str(self.maximum)}"
    def __bool__(self): return self.minimum < self.maximum
    def __len__(self): return (self.maximum - self.minimum).days


class QueryMeta(type):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        return super(QueryMeta, mcs).__new__(mcs, name, bases, attrs)

    def __init__(cls, *args, fields=[], **kwargs):
        cls.__fields__ = getattr(cls, "__fields__", set()) | set(fields)

    def __call__(cls, inquiry, contract, *args, **kwargs):
        fields = {field: kwargs.get(field, None) for field in cls.__fields__}
        instance = super(QueryMeta, cls).__call__(inquiry, contract, fields)
        return instance

class Query(ntuple("Query", "inquiry contract fields"), metaclass=QueryMeta):
    def __str__(self): return f"{self.contract.ticker}|{self.contract.expire.strftime('%Y-%m-%d')}"
    def __getattr__(self, field): return self.fields[field] if field in self.fields.keys() else super().__getattr__(field)

    def __call__(self, *args, **kwargs):
        fields = {key: kwargs.get(key, value) for key, value in self.fields.items()}
        return type(self)(self.inquiry, self.contract, *args, **kwargs | fields)


@total_ordering
class Contract(ntuple("Contract", "ticker expire")):
    def __str__(self): return f"{str(self.ticker).upper()}, {self.expire.strftime('%Y-%m-%d')}"
    def __eq__(self, other): return self.ticker == other.ticker and self.expire == other.expire
    def __lt__(self, other): return self.ticker < other.ticker and self.expire < other.expire


class Variable(object):
    def __str__(self): return "|".join([str(value.name).lower() for value in self if value is not None])
    def __hash__(self): return hash(tuple(self))

    @property
    def title(self): return "|".join([str(string).title() for string in str(self).split("|")])


class Security(Variable, ntuple("Security", "instrument position")): pass
class Strategy(Variable, ntuple("Strategy", "spread instrument position")):
    def __new__(cls, spread, instrument, position, *args, **kwargs): return super().__new__(cls, spread, instrument, position)
    def __init__(self, *args, securities, **kwargs): self.securities = securities


StockLong = Security(Instruments.STOCK, Positions.LONG)
StockShort = Security(Instruments.STOCK, Positions.SHORT)
PutLong = Security(Instruments.PUT, Positions.LONG)
PutShort = Security(Instruments.PUT, Positions.SHORT)
CallLong = Security(Instruments.CALL, Positions.LONG,)
CallShort = Security(Instruments.CALL, Positions.SHORT)
CollarLong = Strategy(Spreads.COLLAR, None, Positions.LONG, securities=[PutLong, CallShort])
CollarShort = Strategy(Spreads.COLLAR, None, Positions.SHORT, securities=[CallLong, PutShort])
VerticalPut = Strategy(Spreads.VERTICAL, Instruments.PUT, None, securities=[PutLong, PutShort])
VerticalCall = Strategy(Spreads.VERTICAL, Instruments.CALL, None, securities=[CallLong, CallShort])


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


class StrategiesMeta(VariablesMeta, variables=[CollarLong, CollarShort, VerticalPut, VerticalCall]):
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



