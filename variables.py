# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 2024
@name:   Finance Variable Objects
@author: Jack Kirby Cook

"""

import pandas as pd
from abc import ABC
from enum import Enum, EnumType
from datetime import date as Date
from functools import total_ordering
from collections import namedtuple as ntuple

from support.dispatchers import typedispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DateRange", "Variables", "Securities", "Strategies"]
__copyright__ = "Copyright 2023,SE Jack Kirby Cook"
__license__ = "MIT License"


class DateRange(ntuple("DateRange", "minimum maximum")):
    def __contains__(self, date): return self.minimum <= date <= self.maximum
    def __new__(cls, dates):
        assert isinstance(dates, list)
        assert all([isinstance(date, Date) for date in dates])
        dates = [date if isinstance(date, Date) else date.date() for date in dates]
        return super().__new__(cls, min(dates), max(dates)) if dates else None

    def __iter__(self): return (date for date in pd.date_range(start=self.minimum, end=self.maximum))
    def __repr__(self): return f"{self.__class__.__name__}({repr(self.minimum)}, {repr(self.maximum)})"
    def __str__(self): return f"{str(self.minimum)}|{str(self.maximum)}"
    def __bool__(self): return self.minimum < self.maximum
    def __len__(self): return (self.maximum - self.minimum).days


@total_ordering
class Symbol(ntuple("Symbol", "ticker")):
    def __str__(self): return str(self.ticker).upper()
    def __eq__(self, other): return str(self.ticker) == str(self.ticker)
    def __lt__(self, other): return str(self.ticker) < str(self.ticker)

@total_ordering
class Contract(ntuple("Contract", "ticker expire")):
    def __str__(self): return "|".join([str(self.ticker).upper(), str(self.expire.strftime("%Y%m%d"))])
    def __eq__(self, other): return str(self.ticker) == str(other.ticker) and self.expire == other.expire
    def __lt__(self, other): return str(self.ticker) < str(other.ticker) and self.expire < other.expire


class Variable(EnumType):
    def __new__(mcs, name, bases, attrs, *args, members=[], **kwargs):
        assert not any([issubclass(base, Enum) for base in bases])
        assert isinstance(members, list) and bool(members)
        def __str__(self): return str(self.name).upper()
        def __int__(self): return int(self.value)
        methods = dict(__str__=__str__, __int__=__int__)
        base = type(name, (object,), methods)
        members = list(map(str.upper, ["empty"] + list(members)))
        members = {member: index for index, member in enumerate(members)}
        for key, value in members.items():
            attrs[key] = value
        return super(Variable, mcs).__new__(mcs, name, (base, Enum), attrs)

    def __call__(cls, variable, *args, **kwargs):
        assert bool(cls._member_map_)
        variable = str(variable).upper() if isinstance(variable, str) else variable
        return super(Variable, cls).__call__(variable, *args, **kwargs)


class MultiVariable(ABC):
    def __int__(self): return int(sum([pow(10, index) * int(value) for index, value in enumerate(reversed(self))]))
    def __str__(self): return str("|".join([str(value) for value in iter(self) if bool(value)]))
    def __hash__(self): return hash(tuple(self.items()))

    def items(self): return list(zip(self.keys(), self.values()))
    def keys(self): return list(self._fields)
    def values(self): return list(self)


class Instruments(members=["STOCK", "OPTION"], metaclass=Variable): pass
class Options(members=["PUT", "CALL"], metaclass=Variable): pass
class Positions(members=["LONG", "SHORT"], metaclass=Variable): pass
class Spreads(members=["STRANGLE", "COLLAR", "VERTICAL"], metaclass=Variable): pass

class Security(MultiVariable, ntuple("Security", "instrument option position")): pass
class Strategy(MultiVariable, ntuple("Strategy", "spread option position")):
    def __new__(cls, spread, option, position, *args, **kwargs): return super().__new__(cls, spread, option, position)
    def __init__(self, *args, options=[], stocks=[], **kwargs): self.options, self.stocks = options, stocks


Status = Enum("Status", ["PROSPECT", "PURCHASED"])
Valuations = Enum("Valuation", ["ARBITRAGE"])
Scenarios = Enum("Scenarios", ["MINIMUM", "MAXIMUM"])
Technicals = Enum("Technicals", ["BARS", "STATISTIC", "STOCHASTIC"])
Querys = Enum("Querys", ["SYMBOL", "CONTRACT"])
Datasets = Enum("Datasets", ["SECURITY", "STRATEGY", "VALUATION", "HOLDINGS", "EXPOSURE"])


print(Instruments.__str__)
print(Instruments.__int__)
print(Instruments.__members__)
print(Technicals.__members__)
raise Exception()


StockLong = Security(Instruments.STOCK, Options.EMPTY, Positions.LONG)
StockShort = Security(Instruments.STOCK, Options.EMPTY, Positions.SHORT)
OptionPutLong = Security(Instruments.OPTION, Options.PUT, Positions.LONG)
OptionPutShort = Security(Instruments.OPTION, Options.PUT, Positions.SHORT)
OptionCallLong = Security(Instruments.OPTION, Options.CALL, Positions.LONG,)
OptionCallShort = Security(Instruments.OPTION, Options.CALL, Positions.SHORT)
VerticalPut = Strategy(Spreads.VERTICAL, Options.PUT, Positions.EMPTY, options=[OptionPutLong, OptionPutShort])
VerticalCall = Strategy(Spreads.VERTICAL, Options.CALL, Positions.EMPTY, options=[OptionCallLong, OptionCallShort])
CollarLong = Strategy(Spreads.COLLAR, Options.EMTPY, Positions.LONG, options=[OptionPutLong, OptionCallShort], stocks=[StockLong])
CollarShort = Strategy(Spreads.COLLAR, Options.EMPTY, Positions.SHORT, options=[OptionCallLong, OptionPutShort], stocks=[StockShort])


class ContextMeta(type):
    def __init_subclass__(mcs, *args, variables, **kwargs): mcs.variables = variables
    def __getitem__(cls, value): return cls.get(value)
    def __iter__(cls): return iter(type(cls).variables)

    @typedispatcher
    def get(cls, key): raise TypeError(type(key).__name__)
    @get.register(int)
    def number(cls, number): return {int(value): value for value in iter(cls)}[int(number)]
    @get.register(tuple)
    def hash(cls, content): return {hash(value): value for value in iter(cls)}[hash(content)]
    @get.register(str)
    def string(cls, string): return {str(value): value for value in iter(cls)}[str(string).lower()]


class SecuritiesMeta(ContextMeta, variables=[StockLong, StockShort, OptionPutLong, OptionPutShort, OptionCallLong, OptionCallShort]):
    def security(cls, instrument, option, position): return Securities[(cls.instrument(instrument), cls.option(option), cls.position(position))]

    @staticmethod
    def instrument(instrument): return Instruments(int(instrument))
    @staticmethod
    def option(option): return Options(int(option))
    @staticmethod
    def position(position): return Positions(int(position))

    @property
    def Stocks(cls): return iter([StockLong, StockShort])
    @property
    def Options(cls): return iter([OptionPutLong, OptionPutShort, OptionCallLong, OptionCallShort])
    @property
    def Puts(cls): return iter([OptionPutLong, OptionPutShort])
    @property
    def Calls(cls): return iter([OptionCallLong, OptionCallShort])


class StrategiesMeta(ContextMeta, variables=[CollarLong, CollarShort, VerticalPut, VerticalCall]):
    def strategy(cls, spread, instrument, position): return Strategies[(cls.spread(spread), cls.instrument(instrument), cls.position(position))]

    @staticmethod
    def spread(spread): return Spreads(int(spread))
    @staticmethod
    def instrument(instrument): return Instruments(int(instrument))
    @staticmethod
    def position(position): return Positions(int(position))

    @property
    def Collars(cls): return iter([CollarLong, CollarShort])
    @property
    def Verticals(cls): return iter([VerticalPut, VerticalCall])


class Securities(object, metaclass=SecuritiesMeta):
    class Stock:
        Long = StockLong
        Short = StockShort
    class Option:
        class Put:
            Long = OptionPutLong
            Short = OptionPutShort
        class Call:
            Long = OptionCallLong
            Short = OptionCallShort

class Strategies(object, metaclass=StrategiesMeta):
    class Collar:
        Long = CollarLong
        Short = CollarShort
    class Vertical:
        Put = VerticalPut
        Call = VerticalCall

class Variables(object):
    Security = Security
    Strategy = Strategy
    Instruments = Instruments
    Options = Options
    Positions = Positions
    Spreads = Spreads
    Valuations = Valuations
    Scenarios = Scenarios
    Technicals = Technicals
    Status = Status
    Querys = Querys
    Datasets = Datasets



